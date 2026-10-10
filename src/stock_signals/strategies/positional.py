"""Positional book v1: 1–6 month momentum holdings.

Entry (weekly, on the last session of the week, filled at the next open):
- Momentum rank ≤ 20 in the universe (score = 50% 6-month + 50% 12-month return ÷ volatility,
  liquidity-filtered), close above SMA200, market not risk-off.

Exits: initial stop 3×ATR (checked daily); weekly close below SMA50; momentum rank
drops below 100 (checked weekly).
"""

from dataclasses import dataclass

import pandas as pd

from stock_signals.models import Candidate, Position
from stock_signals.regime import Regime


@dataclass
class PositionalRules:
    universe: set[str]
    version: str = "positional-v1"
    book: str = "positional"
    min_traded_value: float = 5e7  # ₹5 crore median daily value
    entry_rank: int = 20
    exit_rank: int = 100
    atr_stop: float = 3.0
    entry_band_atr: float = 0.5
    max_new_per_week: int = 5

    def max_new(self, regime: Regime) -> int:
        return self.max_new_per_week if regime.positional_entries_allowed else 0

    def candidates(self, snap: pd.DataFrame, regime: Regime, week_end: bool) -> list[Candidate]:
        if not week_end or not regime.positional_entries_allowed:
            return []
        s = snap[snap.index.isin(self.universe)]
        s = s[
            (s["mom_rank"] <= self.entry_rank)
            & (s["close"] > s["sma200"])
            & (s["traded_value_med20"] >= self.min_traded_value)
            & s["atr14"].gt(0)
        ]
        out = []
        for symbol, r in s.sort_values("mom_rank").iterrows():
            close, atr = float(r["close"]), float(r["atr14"])
            stop = close - self.atr_stop * atr
            if stop <= 0:
                continue
            out.append(
                Candidate(
                    symbol=str(symbol),
                    ref_price=round(close, 2),
                    entry_low=round(close - atr, 2),
                    entry_high=round(close + self.entry_band_atr * atr, 2),
                    stop=round(stop, 2),
                    score=float(r["mom_score"]),
                    reason=(
                        f"momentum rank #{int(r['mom_rank'])}, 6m {r['roc126']:+.0%}, "
                        f"12m {r['roc252']:+.0%}"
                    ),
                )
            )
        return out

    def close_exit(self, pos: Position, row: pd.Series, week_end: bool) -> str | None:
        if not week_end:
            return None
        if pd.notna(row.get("sma50")) and row["close"] < row["sma50"]:
            return "trail_sma50"
        rank = row.get("mom_rank")
        if pd.isna(rank) or rank > self.exit_rank:
            return "rank_exit"
        return None
