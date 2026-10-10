"""Basket backtests and live paper tracking."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
from pymongo import ASCENDING
from pymongo.database import Database

from stock_signals.backtest import BacktestResult
from stock_signals.baskets import (
    BasketBook,
    BasketState,
    BasketStep,
    CoreRules,
    SatelliteRules,
    first_sessions,
)
from stock_signals.costs import CostModel
from stock_signals.metrics import compute_metrics
from stock_signals.panel import Panel, Universe
from stock_signals.regime import regime_on

MAX_CATCH_UP = 30


def build_baskets(
    universe: Universe, core_mode: str = "lowvol"
) -> list[CoreRules | SatelliteRules]:
    return [
        CoreRules(universe=universe.core, mode=core_mode),
        SatelliteRules(universe=universe.swing),
    ]


def split_capital(total: float, core_pct: float) -> dict[str, float]:
    return {"core": total * core_pct / 100, "satellite": total * (100 - core_pct) / 100}


@dataclass
class BasketBacktest:
    sleeves: dict[str, BacktestResult]
    combined: BacktestResult
    nifty500: pd.Series = field(default_factory=pd.Series)


def run_basket_backtest(
    panel: Panel,
    rules: list[CoreRules | SatelliteRules],
    capital: dict[str, float],
    costs: CostModel,
    sector_of: dict[str, str],
    start: str,
    end: str,
    snapshots: dict[str, pd.DataFrame] | None = None,
) -> BasketBacktest:
    snapshots = snapshots if snapshots is not None else panel.by_date()
    months, years = first_sessions(panel.calendar)
    days = [d for d in panel.calendar if start <= d <= end and d in panel.regimes.index]
    sleeves = {}
    for r in rules:
        book = BasketBook(r, BasketState.new(r.name, capital[r.name]), costs, sector_of)
        values = {}
        for d in days:
            values[d] = book.step(d, snapshots[d], regime_on(panel.regimes, d), months, years).value
        eq = pd.Series(values, dtype=float)
        sleeves[r.name] = _result(f"basket-{r.name}", eq, panel, capital[r.name])
    total = sum(s.equity for s in sleeves.values())
    combined = _result("baskets-combined", total, panel, sum(capital.values()))
    n500 = panel.data[panel.data["symbol"] == "NIFTY500"].set_index("date")["close"]
    n500 = n500.reindex(total.index).dropna()
    if len(n500) > 1:
        combined.metrics["nifty500_cagr"] = compute_metrics([], n500 / n500.iloc[0])["cagr"]
        combined.metrics["nifty500_max_drawdown"] = compute_metrics([], n500)["max_drawdown"]
    return BasketBacktest(sleeves=sleeves, combined=combined, nifty500=n500)


def _result(name: str, eq: pd.Series, panel: Panel, capital: float) -> BacktestResult:
    bench = panel.regimes["nifty_close"].reindex(eq.index)
    if len(bench):
        bench = bench / bench.iloc[0] * capital
    r = BacktestResult(
        book="basket",
        variant=name,
        start=eq.index[0] if len(eq) else "",
        end=eq.index[-1] if len(eq) else "",
        capital=capital,
        trades=[],
        equity=eq,
        invested=pd.Series(dtype=float),
        benchmark=bench,
        open_positions=0,
        signals=0,
        skips=0,
    )
    if len(eq):
        r.metrics = compute_metrics([], eq, None, bench)
        for k in (
            "trades",
            "trades_per_month",
            "win_rate",
            "avg_r",
            "avg_win_r",
            "avg_loss_r",
            "profit_factor",
            "avg_days_held",
            "total_costs",
            "expectancy_r",
        ):
            r.metrics.pop(k, None)
    return r


def step_text(name: str, step: BasketStep) -> list[str]:
    lines = []
    if step.executed:
        parts = [
            f"{'BUY' if q > 0 else 'SELL'} {s} {abs(q)} @ {p:,.1f}" for s, q, p in step.executed
        ]
        lines.append(f"🧺 {name.upper()} executed at open (paper): " + ", ".join(parts))
    if step.planned:
        parts = [f"{'BUY' if o.qty > 0 else 'SELL'} {o.symbol} {abs(o.qty)}" for o in step.planned]
        lines.append(
            f"🧺 {name.upper()} rebalance at next open ({step.reason}): " + ", ".join(parts)
        )
    return lines


class BasketPaper:
    def __init__(
        self,
        db: Database,
        rules: list[CoreRules | SatelliteRules],
        capital: dict[str, float],
        costs: CostModel,
        sector_of: dict[str, str],
    ) -> None:
        self.states = db["basket_state"]
        self.events = db["basket_events"]
        self.equity = db["basket_equity"]
        self.rules = rules
        self.capital = capital
        self.costs = costs
        self.sector_of = sector_of

    def load(self, name: str) -> BasketState | None:
        doc = self.states.find_one({"_id": name}, {"_id": 0})
        return BasketState.model_validate(doc) if doc else None

    def all_states(self) -> list[BasketState]:
        return [
            BasketState.model_validate({k: v for k, v in d.items() if k != "_id"})
            for d in self.states.find().sort("_id", ASCENDING)
        ]

    def run(self, latest: str, panel: Panel) -> list[str]:
        self.equity.create_index([("name", ASCENDING), ("date", ASCENDING)], unique=True)
        months, years = first_sessions(panel.calendar)
        lines: list[str] = []
        for r in self.rules:
            state = self.load(r.name) or BasketState.new(r.name, self.capital[r.name])
            todo = [d for d in panel.calendar if d <= latest and d in panel.regimes.index]
            todo = [d for d in todo if d > state.last_date] if state.last_date else todo[-1:]
            todo = todo[-MAX_CATCH_UP:]
            book = BasketBook(r, state, self.costs, self.sector_of)
            for d in todo:
                step = book.step(d, panel.snapshot(d), regime_on(panel.regimes, d), months, years)
                text = step_text(r.name, step)
                if text:
                    self.events.insert_one(
                        {
                            "name": r.name,
                            "date": d,
                            "text": "\n".join(text),
                            "executed": step.executed,
                            "planned": [o.model_dump() for o in step.planned],
                            "created_at": datetime.now(UTC),
                        }
                    )
                    lines += text
                self.equity.update_one(
                    {"name": r.name, "date": d}, {"$set": {"value": step.value}}, upsert=True
                )
            self.states.replace_one({"_id": r.name}, state.model_dump(), upsert=True)
        return lines
