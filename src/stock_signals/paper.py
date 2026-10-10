"""Paper trading: runs each book's BookEngine one session at a time on live EOD data.

Collections
- paper_books   current BookState per variant (derived state, replaced each run)
- paper_events  append-only ledger of every signal, fill, skip, exit
- paper_trades  append-only closed trades
- paper_equity  equity per variant and date (with the Nifty close for comparison)
- bot_settings  pause flag
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
from pymongo import ASCENDING
from pymongo.database import Database

from stock_signals.bars import BarStore
from stock_signals.book import BookEngine, BookState
from stock_signals.costs import CostModel
from stock_signals.models import Event, Trade
from stock_signals.panel import Panel, Universe, load_panel
from stock_signals.regime import Regime, regime_on
from stock_signals.sizing import SizingConfig
from stock_signals.strategies import Strategy

log = logging.getLogger(__name__)

# Sessions replayed at most when the PC was off for a while.
MAX_CATCH_UP = 30


class Paused:
    """Wraps a strategy so it manages exits but issues no new signals."""

    def __init__(self, inner: Strategy) -> None:
        self.inner = inner
        self.book = inner.book
        self.version = inner.version

    def candidates(self, snap, regime, week_end):  # noqa: ANN001 - Strategy protocol
        return []

    def max_new(self, regime: Regime) -> int:
        return 0

    def close_exit(self, pos, row, week_end):  # noqa: ANN001
        return self.inner.close_exit(pos, row, week_end)


@dataclass
class DayReport:
    date: str
    regime: Regime
    events: list[tuple[str, Event]] = field(default_factory=list)  # (variant, event)


@dataclass
class PaperRun:
    days: list[DayReport] = field(default_factory=list)
    note: str = ""

    def text(self, multi_variant_books: set[str] | None = None) -> str:
        if not self.days:
            return self.note or "📭 Nothing new to process"
        multi = multi_variant_books or set()
        out = []
        for day in self.days:
            lines = [f"📅 {day.date}", day.regime.describe()]
            order = ["exit", "partial_exit", "stop_moved", "exit_planned", "fill", "skip", "signal"]
            for kind in order:
                for variant, e in day.events:
                    if e.kind != kind:
                        continue
                    tag = f"[{variant}] " if e.book in multi else ""
                    lines.append(("\n" if kind == "signal" else "") + tag + e.text)
            if len(lines) == 2:
                lines.append("No trades or new signals today.")
            out.append("\n".join(lines))
        if self.note:
            out.insert(0, self.note)
        return "\n\n".join(out)


class PaperStore:
    def __init__(self, db: Database) -> None:
        self.books = db["paper_books"]
        self.events = db["paper_events"]
        self.trades = db["paper_trades"]
        self.equity = db["paper_equity"]
        self.settings = db["bot_settings"]

    def ensure_indexes(self) -> None:
        self.events.create_index([("variant", ASCENDING), ("date", ASCENDING)])
        self.trades.create_index([("variant", ASCENDING), ("exit_date", ASCENDING)])
        self.equity.create_index([("variant", ASCENDING), ("date", ASCENDING)], unique=True)

    def load_state(self, variant: str) -> BookState | None:
        doc = self.books.find_one({"_id": variant}, {"_id": 0})
        return BookState.model_validate(doc) if doc else None

    def save_state(self, state: BookState) -> None:
        self.books.replace_one({"_id": state.variant}, state.model_dump(), upsert=True)

    def states(self) -> list[BookState]:
        return [
            BookState.model_validate({k: v for k, v in d.items() if k != "_id"})
            for d in self.books.find().sort("_id", ASCENDING)
        ]

    def append(
        self,
        variant: str,
        events: list[Event],
        trades: list[Trade],
        date: str,
        equity: float,
        invested: float,
        nifty: float,
    ) -> None:
        now = datetime.now(UTC)
        if events:
            self.events.insert_many(
                [{**e.model_dump(), "variant": variant, "created_at": now} for e in events]
            )
        if trades:
            self.trades.insert_many(
                [{**t.model_dump(), "variant": variant, "created_at": now} for t in trades]
            )
        self.equity.update_one(
            {"variant": variant, "date": date},
            {"$set": {"equity": equity, "invested": invested, "nifty": nifty}},
            upsert=True,
        )

    def is_paused(self) -> bool:
        doc = self.settings.find_one({"_id": "paused"})
        return bool(doc and doc.get("value"))

    def set_paused(self, value: bool) -> None:
        self.settings.update_one({"_id": "paused"}, {"$set": {"value": value}}, upsert=True)

    def trade_list(
        self, variant: str | None = None, book: str | None = None, since: str | None = None
    ) -> list[Trade]:
        q: dict = {}
        if variant:
            q["variant"] = variant
        if book:
            q["book"] = book
        if since:
            q["exit_date"] = {"$gte": since}
        docs = self.trades.find(q, {"_id": 0, "variant": 0, "created_at": 0}).sort(
            "exit_date", ASCENDING
        )
        return [Trade.model_validate(d) for d in docs]

    def equity_frame(self, variant: str) -> pd.DataFrame:
        docs = list(self.equity.find({"variant": variant}, {"_id": 0}).sort("date", ASCENDING))
        return pd.DataFrame(docs).set_index("date") if docs else pd.DataFrame()

    def last_signal(self, symbol: str) -> dict | None:
        return self.events.find_one(
            {"symbol": symbol, "kind": "signal"}, {"_id": 0}, sort=[("date", -1)]
        )


class PaperTrader:
    def __init__(
        self,
        store: PaperStore,
        bars: BarStore,
        universe: Universe,
        strategies: list[Strategy],
        capital: dict[str, float],
        costs: CostModel,
        sizing: SizingConfig,
    ) -> None:
        self.store = store
        self.bars = bars
        self.universe = universe
        self.strategies = strategies
        self.capital = capital
        self.costs = costs
        self.sizing = sizing

    def run(self, latest: str, panel: Panel | None = None) -> PaperRun:
        """Process every session after each book's last processed date, up to `latest`."""
        self.store.ensure_indexes()
        panel = panel or load_panel(self.bars, self.universe, end=latest)
        snapshots: dict[str, pd.DataFrame] = {}
        paused = self.store.is_paused()
        run = PaperRun()
        days: dict[str, DayReport] = {}

        for strat in self.strategies:
            variant = strat.version
            state = self.store.load_state(variant) or BookState.new(
                strat.book, variant, self.capital[strat.book]
            )
            todo = [d for d in panel.calendar if d <= latest and d in panel.regimes.index]
            todo = [d for d in todo if d > state.last_date] if state.last_date else todo[-1:]
            if len(todo) > MAX_CATCH_UP:
                run.note = f"⚠️ {len(todo)} sessions missed; replaying the last {MAX_CATCH_UP} only"
                todo = todo[-MAX_CATCH_UP:]
            engine = BookEngine(
                Paused(strat) if paused else strat,
                state,
                self.costs,
                self.sizing,
                self.universe.sector_of,
            )
            for d in todo:
                if d not in snapshots:
                    snapshots[d] = panel.snapshot(d)
                regime = regime_on(panel.regimes, d)
                res = engine.step(d, snapshots[d], regime, d in panel.week_ends)
                self.store.append(
                    variant, res.events, res.trades, d, res.equity, res.invested, regime.nifty_close
                )
                day = days.setdefault(d, DayReport(date=d, regime=regime))
                day.events += [(variant, e) for e in res.events]
            self.store.save_state(state)

        run.days = [days[d] for d in sorted(days)]
        if paused and run.days:
            run.note = (run.note + "\n" if run.note else "") + (
                "⏸️ Paused: managing open positions only, no new signals (/resume)"
            )
        return run
