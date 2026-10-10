"""Builds the indicator panel and per-date context the strategies run on."""

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from stock_signals import dates
from stock_signals.bars import BarStore
from stock_signals.indicators import add_indicators, add_momentum
from stock_signals.regime import regime_table
from stock_signals.universe import CALENDAR_SYMBOL, MARKET_SYMBOLS, SymbolStore

# Calendar days of history loaded for a single-day scan (≥ 253 sessions for 12-month momentum).
SCAN_LOOKBACK_DAYS = 600


@dataclass
class Universe:
    swing: set[str]
    positional: set[str]
    sector_of: dict[str, str]
    core: set[str] = field(default_factory=set)

    @property
    def all(self) -> set[str]:
        return self.swing | self.positional | self.core

    @classmethod
    def from_store(cls, symbols: SymbolStore) -> "Universe":
        docs = list(symbols.col.find({"status": "active", "kind": "equity"}, {"_id": 0}))
        return cls(
            swing={d["symbol"] for d in docs if "NIFTY200" in d.get("indices", [])},
            positional={d["symbol"] for d in docs if "NIFTY500" in d.get("indices", [])},
            sector_of={d["symbol"]: d.get("industry", "") for d in docs},
            core={d["symbol"] for d in docs if "NIFTY100" in d.get("indices", [])},
        )


@dataclass
class Panel:
    """Indicators for every (symbol, date) plus the regime table and calendar."""

    data: pd.DataFrame
    regimes: pd.DataFrame
    calendar: list[str]
    week_ends: set[str]

    def snapshot(self, date: str) -> pd.DataFrame:
        return self.data[self.data["date"] == date].set_index("symbol")

    def by_date(self) -> dict[str, pd.DataFrame]:
        return {d: f.set_index("symbol") for d, f in self.data.groupby("date", sort=True)}


def week_end_dates(calendar: list[str]) -> set[str]:
    """Last trading session of each ISO week. The latest week counts only if it is complete
    (a Friday), so a mid-week run never pretends the week has ended."""
    if not calendar:
        return set()
    s = pd.Series(pd.to_datetime(calendar), index=calendar)
    iso = s.dt.isocalendar()
    keys = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    ends = set(s.groupby(keys.values).apply(lambda x: x.index[-1]))
    last = calendar[-1]
    if last in ends and pd.Timestamp(last).weekday() < 4:
        ends.discard(last)
    return ends


def build_panel(bars: pd.DataFrame, universe: Universe, min_traded_value: float = 5e7) -> Panel:
    data = add_indicators(bars)
    eligible = data["symbol"].isin(universe.positional)
    data = add_momentum(data, eligible=eligible, min_traded_value=min_traded_value)
    regimes = regime_table(data, CALENDAR_SYMBOL, universe.all)
    calendar = sorted(data.loc[data["symbol"] == CALENDAR_SYMBOL, "date"].unique())
    return Panel(data=data, regimes=regimes, calendar=calendar, week_ends=week_end_dates(calendar))


def load_panel(
    store: BarStore,
    universe: Universe,
    end: str,
    start: str | None = None,
    lookback_days: int = SCAN_LOOKBACK_DAYS,
) -> Panel:
    if start is None:
        start = dates.to_str(dates.parse(end) - timedelta(days=lookback_days))
    symbols = sorted(universe.all | set(MARKET_SYMBOLS))
    bars = store.load(symbols, start=start, end=end)
    return build_panel(bars, universe)
