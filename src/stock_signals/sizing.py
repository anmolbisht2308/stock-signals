"""Position sizing and portfolio limits."""

import math
from dataclasses import dataclass

from stock_signals.models import Candidate, Signal


@dataclass(frozen=True)
class SizingConfig:
    risk_pct: float = 1.0  # % of book equity risked per trade (entry − stop)
    max_position_pct: float = 20.0  # % of book equity in one stock
    max_per_sector: int = 3
    max_positions: int = 10


def position_qty(equity: float, entry: float, stop: float, cfg: SizingConfig) -> int:
    risk_per_share = entry - stop
    if risk_per_share <= 0 or entry <= 0:
        return 0
    by_risk = math.floor(equity * cfg.risk_pct / 100 / risk_per_share)
    by_cap = math.floor(equity * cfg.max_position_pct / 100 / entry)
    return max(min(by_risk, by_cap), 0)


def select_signals(
    candidates: list[Candidate],
    *,
    book: str,
    date: str,
    version: str,
    equity: float,
    free_cash: float,
    held: set[str],
    sector_of: dict[str, str],
    sector_counts: dict[str, int],
    open_count: int,
    max_new: int,
    cfg: SizingConfig,
) -> list[Signal]:
    """Size candidates (best first) and keep those within position, sector and cash limits."""
    out: list[Signal] = []
    counts = dict(sector_counts)
    cash = free_cash
    for c in candidates:
        if len(out) >= max_new or open_count + len(out) >= cfg.max_positions:
            break
        if c.symbol in held:
            continue
        sector = sector_of.get(c.symbol, "")
        if sector and counts.get(sector, 0) >= cfg.max_per_sector:
            continue
        qty = position_qty(equity, c.ref_price, c.stop, cfg)
        cost = qty * c.entry_high
        if qty <= 0 or cost > cash:
            continue
        risk = c.ref_price - c.stop
        out.append(
            Signal(
                **c.model_dump(),
                id=Signal.make_id(book, c.symbol, date, version),
                book=book,  # type: ignore[arg-type]
                date=date,
                qty=qty,
                risk_per_share=round(risk, 2),
                risk_amount=round(risk * qty, 2),
                rule_version=version,
                sector=sector,
            )
        )
        cash -= cost
        if sector:
            counts[sector] = counts.get(sector, 0) + 1
    return out
