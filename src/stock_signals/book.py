"""Position engine for one book. Used by both the backtester and the live paper trader,
so paper results follow exactly the rules that were backtested.

Each `step(date)` processes one session in this order:

1. Open: execute exits decided on the previous close, then fill yesterday's signals at the
   open (skipped if the open gapped above the entry zone or at/below the stop).
2. Intraday: stop first (if stop and target are both inside the day's range, the stop is
   assumed to hit first); T1 books half and moves the stop to cost; T2 exits the rest.
   A gap through a level fills at the open.
3. Close: strategy exits (time stop, trailing, rank) are decided and queued for the next open.
4. New signals for the next session are generated and sized against current equity.
"""

import math

import pandas as pd
from pydantic import BaseModel, Field

from stock_signals import messages
from stock_signals.costs import CostModel
from stock_signals.models import Event, Leg, Position, Signal, Trade
from stock_signals.regime import Regime
from stock_signals.sizing import SizingConfig, select_signals
from stock_signals.strategies import Strategy


class BookState(BaseModel):
    book: str
    variant: str  # rule/model version this book trades, e.g. "swing-v1"
    capital: float
    cash: float
    positions: list[Position] = Field(default_factory=list)
    pending: list[Signal] = Field(default_factory=list)
    last_date: str | None = None

    @classmethod
    def new(cls, book: str, variant: str, capital: float) -> "BookState":
        return cls(book=book, variant=variant, capital=capital, cash=capital)

    def invested(self) -> float:
        return sum(p.qty * p.last_close for p in self.positions)

    def equity(self) -> float:
        return self.cash + self.invested()


class StepResult(BaseModel):
    date: str
    events: list[Event] = Field(default_factory=list)
    trades: list[Trade] = Field(default_factory=list)
    equity: float = 0.0
    invested: float = 0.0


class BookEngine:
    def __init__(
        self,
        strategy: Strategy,
        state: BookState,
        costs: CostModel,
        sizing: SizingConfig,
        sector_of: dict[str, str],
    ) -> None:
        self.strategy = strategy
        self.state = state
        self.costs = costs
        self.sizing = sizing
        self.sector_of = sector_of

    # ---- helpers -------------------------------------------------------------

    def _event(self, res: StepResult, kind: str, symbol: str, text: str, **data) -> None:
        res.events.append(
            Event(
                date=res.date, book=self.state.book, kind=kind, symbol=symbol, text=text, data=data
            )  # type: ignore[arg-type]
        )

    def _sell(
        self, res: StepResult, pos: Position, raw_price: float, qty: int, reason: str
    ) -> None:
        fill, charges = self.costs.sell(raw_price, qty)
        self.state.cash += fill * qty - charges
        pos.legs.append(Leg(date=res.date, price=fill, qty=qty, reason=reason, costs=charges))
        pos.qty -= qty
        if pos.qty > 0:
            self._event(
                res,
                "partial_exit",
                pos.symbol,
                messages.partial_text(pos, fill, qty, reason),
                price=fill,
                qty=qty,
                reason=reason,
            )
            return
        trade = self._close(pos, res.date)
        res.trades.append(trade)
        self.state.positions = [p for p in self.state.positions if p.id != pos.id]
        self._event(
            res,
            "exit",
            pos.symbol,
            messages.exit_text(trade, fill),
            **trade.model_dump(include={"net_pnl", "r_multiple", "exit_reason"}),
        )

    def _close(self, pos: Position, date: str) -> Trade:
        exit_value = sum(leg.price * leg.qty for leg in pos.legs)
        gross = exit_value - pos.entry_price * pos.initial_qty
        costs = pos.buy_costs + sum(leg.costs for leg in pos.legs)
        net = gross - costs
        return Trade(
            position_id=pos.id,
            book=pos.book,
            symbol=pos.symbol,
            sector=pos.sector,
            rule_version=pos.rule_version,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=date,
            avg_exit_price=exit_value / pos.initial_qty,
            qty=pos.initial_qty,
            gross_pnl=round(gross, 2),
            costs=round(costs, 2),
            net_pnl=round(net, 2),
            r_multiple=round(net / pos.initial_risk, 3),
            days_held=pos.bars_held,
            exit_reason=pos.legs[-1].reason,
        )

    # ---- the session ---------------------------------------------------------

    def step(self, date: str, snap: pd.DataFrame, regime: Regime, week_end: bool) -> StepResult:
        res = StepResult(date=date)
        self._open(res, snap)
        self._intraday(res, snap)
        self._close_phase(res, snap, week_end)
        self._generate(res, snap, regime, week_end)
        self.state.last_date = date
        res.equity = self.state.equity()
        res.invested = self.state.invested()
        return res

    def _open(self, res: StepResult, snap: pd.DataFrame) -> None:
        for pos in list(self.state.positions):
            if pos.pending_exit and pos.symbol in snap.index:
                self._sell(res, pos, float(snap.at[pos.symbol, "open"]), pos.qty, pos.pending_exit)

        held = {p.symbol for p in self.state.positions}
        for sig in self.state.pending:
            if sig.symbol not in snap.index:
                self._event(res, "skip", sig.symbol, messages.skip_text(sig, "no trade today"))
                continue
            if sig.symbol in held:
                continue
            open_ = float(snap.at[sig.symbol, "open"])
            if open_ > sig.entry_high:
                self._event(
                    res,
                    "skip",
                    sig.symbol,
                    messages.skip_text(sig, f"opened {messages.px(open_)} above entry"),
                )
                continue
            if open_ <= sig.stop:
                self._event(
                    res,
                    "skip",
                    sig.symbol,
                    messages.skip_text(sig, f"opened {messages.px(open_)} below stop"),
                )
                continue
            fill, _ = self.costs.buy(open_, 1)
            qty = sig.qty
            per_share = fill * (1 + 0.002)  # headroom for charges
            if qty * per_share > self.state.cash:
                qty = math.floor(self.state.cash / per_share)
            if qty <= 0:
                self._event(res, "skip", sig.symbol, messages.skip_text(sig, "not enough cash"))
                continue
            fill, charges = self.costs.buy(open_, qty)
            self.state.cash -= fill * qty + charges
            pos = Position(
                id=sig.id,
                book=sig.book,
                symbol=sig.symbol,
                sector=sig.sector,
                signal_id=sig.id,
                rule_version=sig.rule_version,
                entry_date=res.date,
                entry_price=fill,
                initial_qty=qty,
                qty=qty,
                stop=sig.stop,
                initial_stop=sig.stop,
                t1=sig.t1,
                t2=sig.t2,
                time_stop_days=sig.time_stop_days,
                buy_costs=charges,
                last_close=fill,
            )
            self.state.positions.append(pos)
            held.add(sig.symbol)
            self._event(res, "fill", sig.symbol, messages.fill_text(pos), price=fill, qty=qty)
        self.state.pending = []

    def _intraday(self, res: StepResult, snap: pd.DataFrame) -> None:
        for pos in list(self.state.positions):
            if pos.symbol not in snap.index or pos.qty <= 0:
                continue
            bar = snap.loc[pos.symbol]
            open_, high, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
            if low <= pos.stop:
                reason = "breakeven_stop" if pos.t1_done else "stop"
                self._sell(res, pos, min(open_, pos.stop), pos.qty, reason)
                continue
            if pos.t1 and not pos.t1_done and high >= pos.t1:
                half = pos.qty // 2 or pos.qty
                pos.t1_done = True
                self._sell(res, pos, max(open_, pos.t1), half, "t1")
                if pos.qty == 0:
                    continue
                if pos.stop < pos.entry_price:
                    pos.stop = round(pos.entry_price, 2)
                    self._event(
                        res, "stop_moved", pos.symbol, messages.stop_moved_text(pos), stop=pos.stop
                    )
            if pos.t1_done and pos.t2 and high >= pos.t2 and pos.qty > 0:
                self._sell(res, pos, max(open_, pos.t2), pos.qty, "t2")

    def _close_phase(self, res: StepResult, snap: pd.DataFrame, week_end: bool) -> None:
        for pos in self.state.positions:
            if pos.symbol not in snap.index:
                continue
            row = snap.loc[pos.symbol]
            pos.last_close = float(row["close"])
            pos.bars_held += 1
            if pos.pending_exit:
                continue
            reason = self.strategy.close_exit(pos, row, week_end)
            if reason:
                pos.pending_exit = reason
                self._event(
                    res,
                    "exit_planned",
                    pos.symbol,
                    messages.exit_planned_text(pos, reason),
                    reason=reason,
                )

    def _generate(
        self, res: StepResult, snap: pd.DataFrame, regime: Regime, week_end: bool
    ) -> None:
        positions = self.state.positions
        sector_counts: dict[str, int] = {}
        for p in positions:
            if p.sector:
                sector_counts[p.sector] = sector_counts.get(p.sector, 0) + 1
        signals = select_signals(
            self.strategy.candidates(snap, regime, week_end),
            book=self.state.book,
            date=res.date,
            version=self.state.variant,
            equity=self.state.equity(),
            free_cash=self.state.cash,
            held={p.symbol for p in positions},
            sector_of=self.sector_of,
            sector_counts=sector_counts,
            open_count=len(positions),
            max_new=self.strategy.max_new(regime),
            cfg=self.sizing,
        )
        self.state.pending = signals
        for s in signals:
            self._event(res, "signal", s.symbol, messages.signal_text(s), signal=s.model_dump())
