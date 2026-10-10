"""Daily data pipeline: backfill history, incremental update, splits, quality, liquidity."""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from stock_signals import dates
from stock_signals.bars import BarStore
from stock_signals.liquidity import median_traded_value
from stock_signals.quality import Issue, QualityReport, check_frame
from stock_signals.sources import FetchResult, PriceSource
from stock_signals.universe import CALENDAR_SYMBOL, MARKET_SYMBOLS, SymbolStore

log = logging.getLogger(__name__)

# Re-download this many days before the last stored bar to pick up late corrections.
OVERLAP_DAYS = 7


class PipelineError(RuntimeError):
    pass


@dataclass
class RunResult:
    kind: str
    end: str
    symbols: int = 0
    rows: int = 0
    failed: list[str] = field(default_factory=list)
    refetched: list[str] = field(default_factory=list)
    quality: QualityReport = field(default_factory=QualityReport)
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"📥 Data {self.kind} up to {self.end}: {self.symbols} symbols, "
            f"{self.rows:,} rows in {self.seconds:.0f}s"
        ]
        if self.failed:
            lines.append(f"❌ Failed ({len(self.failed)}): {', '.join(self.failed[:20])}")
        if self.refetched:
            lines.append(f"🔁 Split/bonus refetch: {', '.join(self.refetched)}")
        lines.append(self.quality.summary())
        return "\n".join(lines)


class Pipeline:
    def __init__(
        self,
        bars: BarStore,
        symbols: SymbolStore,
        source: PriceSource,
        history_years: int = 10,
        batch_size: int = 50,
    ) -> None:
        self.bars = bars
        self.symbols = symbols
        self.source = source
        self.history_years = history_years
        self.batch_size = batch_size

    def setup(self) -> None:
        self.bars.ensure_indexes()
        self.symbols.ensure_indexes()
        self.symbols.ensure_market_symbols()

    def _targets(self, only: list[str] | None) -> list[str]:
        equities = only if only is not None else self.symbols.active(kind="equity")
        # Market series first: the Nifty 50 bars are the trading calendar for quality checks.
        return list(MARKET_SYMBOLS) + [s for s in equities if s not in MARKET_SYMBOLS]

    def _fetch(self, symbols: list[str], start: date, end: date) -> FetchResult:
        if not symbols or start > end:
            return FetchResult()
        res = self.source.fetch(symbols, start, end)
        b = res.bars
        if not b.empty:
            res.bars = b[(b["date"] >= dates.to_str(start)) & (b["date"] <= dates.to_str(end))]
        return res

    def run(
        self,
        full: bool = False,
        only: list[str] | None = None,
        now: datetime | None = None,
    ) -> RunResult:
        """`full=True` re-downloads the whole history; otherwise only what's new."""
        started = time.monotonic()
        end = dates.last_complete_session(now)
        history_start = end - timedelta(days=365 * self.history_years)
        result = RunResult(kind="backfill" if full else "update", end=dates.to_str(end))
        last = {} if full else self.bars.last_dates()
        targets = self._targets(only)
        no_volume = set(MARKET_SYMBOLS)
        calendar: list[str] = []

        for i in range(0, len(targets), self.batch_size):
            batch = targets[i : i + self.batch_size]
            new = [s for s in batch if s not in last]
            existing = [s for s in batch if s in last]
            fetched = [self._fetch(new, history_start, end)]
            window_start = history_start
            if existing:
                window_start = dates.parse(min(last[s] for s in existing)) - timedelta(
                    days=OVERLAP_DAYS
                )
                fetched.append(self._fetch(existing, window_start, end))

            frames = [f.bars for f in fetched if not f.bars.empty]
            batch_bars = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            splits = pd.concat([f.splits for f in fetched], ignore_index=True)
            failed = [s for f in fetched for s in f.failed]
            result.failed += failed
            result.quality.issues += [Issue("fetch_failed", s, "no data returned") for s in failed]

            if not batch_bars.empty:
                result.rows += self.bars.upsert(batch_bars, self.source.name)
                result.symbols += batch_bars["symbol"].nunique()

            if not calendar:
                calendar = self.bars.dates(CALENDAR_SYMBOL, start=dates.to_str(history_start))
                if not calendar:
                    raise PipelineError(f"No {CALENDAR_SYMBOL} bars; cannot build the calendar")

            # A split newer than our stored history means the stored prices are unadjusted.
            for row in splits.itertuples(index=False):
                if row.symbol in last and row.date > last[row.symbol]:
                    if row.symbol not in result.refetched:
                        result.refetched.append(row.symbol)
            for symbol in result.refetched:
                if symbol in batch:
                    self.bars.delete_symbol(symbol)
                    res = self._fetch([symbol], history_start, end)
                    self.bars.upsert(res.bars, self.source.name)

            if not batch_bars.empty:
                result.quality.extend(
                    check_frame(batch_bars, calendar, splits, no_volume_symbols=no_volume)
                )

        self._update_liquidity(targets, end)
        result.seconds = time.monotonic() - started
        return result

    def _update_liquidity(self, symbols: list[str], end: date) -> None:
        equities = [s for s in symbols if s not in MARKET_SYMBOLS]
        start = dates.to_str(end - timedelta(days=45))
        for i in range(0, len(equities), self.batch_size):
            batch = equities[i : i + self.batch_size]
            recent = self.bars.load(batch, start=start)
            values = median_traded_value(recent)
            last_dates = recent.groupby("symbol")["date"].max().to_dict() if len(recent) else {}
            for symbol in batch:
                self.symbols.set_fields(
                    symbol,
                    {
                        "median_traded_value_20d": values.get(symbol),
                        "last_bar_date": last_dates.get(symbol),
                    },
                )
