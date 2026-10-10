"""Daily event-driven backtest using the same BookEngine as paper trading."""

from dataclasses import dataclass, field

import pandas as pd

from stock_signals.book import BookEngine, BookState
from stock_signals.costs import CostModel
from stock_signals.metrics import compute_metrics
from stock_signals.models import Trade
from stock_signals.panel import Panel
from stock_signals.regime import regime_on
from stock_signals.sizing import SizingConfig
from stock_signals.strategies import Strategy

# Walk-forward periods. The final period is for one last check only — don't tune on it.
PERIODS = {
    "tune": ("2014-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2024-12-31"),
    "final": ("2025-01-01", "2099-12-31"),
}


@dataclass
class BacktestResult:
    book: str
    variant: str
    start: str
    end: str
    capital: float
    trades: list[Trade]
    equity: pd.Series
    invested: pd.Series
    benchmark: pd.Series
    open_positions: int
    signals: int
    skips: int
    metrics: dict[str, float] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.model_dump() for t in self.trades])


def run_backtest(
    panel: Panel,
    strategy: Strategy,
    capital: float,
    costs: CostModel,
    sizing: SizingConfig,
    sector_of: dict[str, str],
    start: str,
    end: str,
    snapshots: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    snapshots = snapshots if snapshots is not None else panel.by_date()
    state = BookState.new(strategy.book, strategy.version, capital)
    engine = BookEngine(strategy, state, costs, sizing, sector_of)
    days = [d for d in panel.calendar if start <= d <= end and d in panel.regimes.index]
    equity, invested, trades = {}, {}, []
    signals = skips = 0
    for d in days:
        res = engine.step(d, snapshots[d], regime_on(panel.regimes, d), d in panel.week_ends)
        equity[d], invested[d] = res.equity, res.invested
        trades += res.trades
        signals += sum(e.kind == "signal" for e in res.events)
        skips += sum(e.kind == "skip" for e in res.events)

    eq, inv = pd.Series(equity, dtype=float), pd.Series(invested, dtype=float)
    bench = panel.regimes["nifty_close"].reindex(eq.index)
    if len(bench):
        bench = bench / bench.iloc[0] * capital
    result = BacktestResult(
        book=strategy.book,
        variant=strategy.version,
        start=days[0] if days else start,
        end=days[-1] if days else end,
        capital=capital,
        trades=trades,
        equity=eq,
        invested=inv,
        benchmark=bench,
        open_positions=len(state.positions),
        signals=signals,
        skips=skips,
    )
    if len(eq):
        result.metrics = compute_metrics(trades, eq, inv, bench)
    return result
