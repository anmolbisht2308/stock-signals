"""Core + satellite baskets.

Core (long-term): the 20 lowest-volatility Nifty 100 stocks, equal weight (or the
NIFTYBEES ETF with `mode="etf"`). New selection in the first session of each year;
in between, the first session of each month restores the weights if any drifted by
more than 5 percentage points.

Satellite (momentum): the top 30 Nifty 200 stocks by momentum score (Nifty200
Momentum 30 style), weighted by score, capped at 10% per stock and 30% per sector.
New selection in the first session of June and December. Regime overlay: on the first
session of each month the sleeve is cut to 50% invested when the market is risk-off
and restored when it is not.

Decisions are taken on the close and executed at the next open, like the trade books.
"""

import math
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, Field

from stock_signals.costs import CostModel
from stock_signals.regime import Regime


def first_sessions(calendar: list[str]) -> tuple[set[str], set[str]]:
    """(first session of each month, first session of each year)."""
    months, years = set(), set()
    prev = None
    for d in calendar:
        if prev is None or d[:7] != prev[:7]:
            months.add(d)
        if prev is None or d[:4] != prev[:4]:
            years.add(d)
        prev = d
    return months, years


def cap_weights(
    raw: dict[str, float],
    cap: float,
    sector_of: dict[str, str] | None = None,
    sector_cap: float | None = None,
    iterations: int = 50,
) -> dict[str, float]:
    """Normalise positive weights to 1, then iteratively clip stock and sector caps and
    redistribute the excess to the names still below their caps."""
    w = {k: v for k, v in raw.items() if v > 0}
    if not w:
        return {}
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    sector_of = sector_of or {}
    for _ in range(iterations):
        frozen: set[str] = set()
        for k, v in w.items():
            if v > cap + 1e-12:
                w[k] = cap
                frozen.add(k)
            elif v >= cap - 1e-12:
                frozen.add(k)
        if sector_cap:
            sectors: dict[str, float] = {}
            for k, v in w.items():
                s = sector_of.get(k, "")
                if s:
                    sectors[s] = sectors.get(s, 0.0) + v
            for s, tot in sectors.items():
                if tot > sector_cap + 1e-12:
                    scale = sector_cap / tot
                    for k in w:
                        if sector_of.get(k) == s:
                            w[k] *= scale
                            frozen.add(k)
                elif tot >= sector_cap - 1e-12:
                    frozen.update(k for k in w if sector_of.get(k) == s)
        excess = 1.0 - sum(w.values())
        free = {k: v for k, v in w.items() if k not in frozen}
        if excess <= 1e-9 or not free:
            break
        free_total = sum(free.values())
        for k, v in free.items():
            w[k] = v + excess * v / free_total
    return w


@dataclass
class CoreRules:
    universe: set[str]
    name: str = "core"
    mode: str = "lowvol"  # or "etf"
    etf: str = "NIFTYBEES"
    size: int = 20
    drift_pts: float = 5.0

    def select(self, snap: pd.DataFrame, sector_of: dict[str, str]) -> dict[str, float]:
        if self.mode == "etf":
            return {self.etf: 1.0} if self.etf in snap.index else {}
        s = snap[snap.index.isin(self.universe) & snap["vol252"].notna()]
        s = s[s["bars_count"] >= 252]
        picks = s.sort_values("vol252").head(self.size).index
        return cap_weights({str(k): 1.0 for k in picks}, cap=1.0)

    def is_selection_day(self, date: str, months: set[str], years: set[str]) -> bool:
        return date in years

    def is_review_day(self, date: str, months: set[str]) -> bool:
        return date in months

    def invest_fraction(self, regime: Regime) -> float:
        return 1.0


@dataclass
class SatelliteRules:
    universe: set[str]
    name: str = "satellite"
    size: int = 30
    stock_cap: float = 0.10
    sector_cap: float = 0.30
    rebalance_months: tuple[str, ...] = ("06", "12")
    risk_off_fraction: float = 0.5
    drift_pts: float = 100.0  # no drift rebalance between selections

    def select(self, snap: pd.DataFrame, sector_of: dict[str, str]) -> dict[str, float]:
        s = snap[snap.index.isin(self.universe) & (snap["mom_score"] > 0)]
        top = s.sort_values("mom_score", ascending=False).head(self.size)
        raw = {str(k): float(v) for k, v in top["mom_score"].items()}
        return cap_weights(raw, self.stock_cap, sector_of, self.sector_cap)

    def is_selection_day(self, date: str, months: set[str], years: set[str]) -> bool:
        return date in months and date[5:7] in self.rebalance_months

    def is_review_day(self, date: str, months: set[str]) -> bool:
        return date in months

    def invest_fraction(self, regime: Regime) -> float:
        return self.risk_off_fraction if regime.mode == "risk_off" else 1.0


class Order(BaseModel):
    symbol: str
    qty: int  # + buy, − sell


class BasketState(BaseModel):
    name: str
    capital: float
    cash: float
    holdings: dict[str, int] = Field(default_factory=dict)
    prices: dict[str, float] = Field(default_factory=dict)
    targets: dict[str, float] = Field(default_factory=dict)
    invest_fraction: float = 1.0
    pending: list[Order] = Field(default_factory=list)
    last_selection: str | None = None
    last_date: str | None = None
    costs_paid: float = 0.0

    @classmethod
    def new(cls, name: str, capital: float) -> "BasketState":
        return cls(name=name, capital=capital, cash=capital)

    def value(self) -> float:
        return self.cash + sum(q * self.prices.get(s, 0.0) for s, q in self.holdings.items())

    def weights(self) -> dict[str, float]:
        total = self.value()
        if total <= 0:
            return {}
        return {s: q * self.prices.get(s, 0.0) / total for s, q in self.holdings.items()}


@dataclass
class BasketStep:
    date: str
    value: float
    executed: list[tuple[str, int, float]] = field(default_factory=list)  # symbol, qty, price
    planned: list[Order] = field(default_factory=list)
    reason: str = ""


class BasketBook:
    def __init__(
        self,
        rules: CoreRules | SatelliteRules,
        state: BasketState,
        costs: CostModel,
        sector_of: dict[str, str],
    ) -> None:
        self.rules = rules
        self.state = state
        self.costs = costs
        self.sector_of = sector_of

    def step(
        self, date: str, snap: pd.DataFrame, regime: Regime, months: set[str], years: set[str]
    ) -> BasketStep:
        st = self.state
        out = BasketStep(date=date, value=0.0)
        self._execute(out, snap)
        for sym in st.holdings:
            if sym in snap.index:
                st.prices[sym] = float(snap.at[sym, "close"])

        reason = ""
        if not st.targets or self.rules.is_selection_day(date, months, years):
            targets = self.rules.select(snap, self.sector_of)
            if targets:
                st.targets, st.last_selection = targets, date
                reason = "new selection"
        if not reason and self.rules.is_review_day(date, months):
            fraction = self.rules.invest_fraction(regime)
            if fraction != st.invest_fraction:
                reason = "risk-off: cut to 50%" if fraction < 1 else "risk-on: back to 100%"
            elif self._max_drift() > self.rules.drift_pts / 100:
                reason = "weights drifted"
        if reason:
            st.invest_fraction = self.rules.invest_fraction(regime)
            st.pending = self._orders(snap)
            out.planned, out.reason = st.pending, reason
        st.last_date = date
        out.value = st.value()
        return out

    def _max_drift(self) -> float:
        current = self.state.weights()
        f = self.state.invest_fraction
        names = set(current) | set(self.state.targets)
        if not names:
            return 0.0
        return max(abs(current.get(s, 0.0) - f * self.state.targets.get(s, 0.0)) for s in names)

    def _orders(self, snap: pd.DataFrame) -> list[Order]:
        st = self.state
        total = st.value()
        orders = []
        for sym in sorted(set(st.targets) | set(st.holdings)):
            if sym not in snap.index:
                continue
            price = float(snap.at[sym, "close"])
            target_qty = math.floor(total * st.invest_fraction * st.targets.get(sym, 0.0) / price)
            diff = target_qty - st.holdings.get(sym, 0)
            if diff:
                orders.append(Order(symbol=sym, qty=diff))
        return orders

    def _execute(self, out: BasketStep, snap: pd.DataFrame) -> None:
        st = self.state
        if not st.pending:
            return
        sells = [o for o in st.pending if o.qty < 0]
        buys = [o for o in st.pending if o.qty > 0]
        for o in sells:
            if o.symbol not in snap.index:
                continue
            fill, charges = self.costs.sell(float(snap.at[o.symbol, "open"]), -o.qty)
            st.cash += fill * -o.qty - charges
            st.costs_paid += charges
            st.holdings[o.symbol] = st.holdings.get(o.symbol, 0) + o.qty
            if st.holdings[o.symbol] <= 0:
                st.holdings.pop(o.symbol)
                st.prices.pop(o.symbol, None)
            out.executed.append((o.symbol, o.qty, fill))
        for o in buys:
            if o.symbol not in snap.index:
                continue
            open_ = float(snap.at[o.symbol, "open"])
            qty = o.qty
            fill, charges = self.costs.buy(open_, qty)
            if fill * qty + charges > st.cash:
                qty = min(qty, math.floor(st.cash / (fill * 1.002)))
                fill, charges = self.costs.buy(open_, max(qty, 1))
                while qty > 0 and fill * qty + charges > st.cash:
                    qty -= 1
                    fill, charges = self.costs.buy(open_, max(qty, 1))
            if qty <= 0:
                continue
            st.cash -= fill * qty + charges
            st.costs_paid += charges
            st.holdings[o.symbol] = st.holdings.get(o.symbol, 0) + qty
            st.prices[o.symbol] = fill
            out.executed.append((o.symbol, qty, fill))
        st.pending = []
