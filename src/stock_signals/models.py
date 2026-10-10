"""Signals, positions, closed trades and ledger events.

Prices are rupees (floats, rounded to paise when shown). Dates are IST `YYYY-MM-DD`.
"""

from typing import Literal

from pydantic import BaseModel, Field

Book = Literal["swing", "positional"]


class Candidate(BaseModel):
    """A rule hit before sizing and portfolio limits."""

    symbol: str
    ref_price: float
    entry_low: float
    entry_high: float
    stop: float
    t1: float | None = None
    t2: float | None = None
    time_stop_days: int | None = None
    score: float = 0.0
    reason: str = ""


class Signal(Candidate):
    id: str
    book: Book
    date: str
    qty: int
    risk_per_share: float
    risk_amount: float
    rule_version: str
    sector: str = ""

    @staticmethod
    def make_id(book: str, symbol: str, date: str, version: str) -> str:
        return f"{book}:{version}:{symbol}:{date}"


class Leg(BaseModel):
    date: str
    price: float
    qty: int
    reason: str
    costs: float


class Position(BaseModel):
    id: str
    book: Book
    symbol: str
    sector: str = ""
    signal_id: str
    rule_version: str
    entry_date: str
    entry_price: float
    initial_qty: int
    qty: int
    stop: float
    initial_stop: float
    t1: float | None = None
    t2: float | None = None
    t1_done: bool = False
    time_stop_days: int | None = None
    bars_held: int = 0
    pending_exit: str | None = None
    buy_costs: float = 0.0
    legs: list[Leg] = Field(default_factory=list)
    last_close: float = 0.0

    @property
    def initial_risk(self) -> float:
        return max(self.entry_price - self.initial_stop, 0.01) * self.initial_qty


class Trade(BaseModel):
    """A fully closed position."""

    position_id: str
    book: Book
    symbol: str
    sector: str = ""
    rule_version: str
    entry_date: str
    entry_price: float
    exit_date: str
    avg_exit_price: float
    qty: int
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float
    days_held: int
    exit_reason: str


class Event(BaseModel):
    """Append-only ledger entry; also the source of Telegram messages."""

    date: str
    book: Book
    kind: Literal["signal", "fill", "skip", "partial_exit", "exit", "stop_moved", "exit_planned"]
    symbol: str
    text: str
    data: dict = Field(default_factory=dict)
