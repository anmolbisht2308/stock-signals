"""Swing book v1: 3–20 day trades in liquid uptrending stocks.

Entry (on the close, filled at the next open):
- Trend: close > SMA50 > SMA200
- Setup A, breakout: close above the previous 20-day high, volume ≥ 1.5× its 20-day average
- Setup B, pullback: low touched EMA20 in the last 3 days, close back above EMA20 and
  above yesterday's close, within 1 ATR of EMA20
- RSI(14) below 80 (not overextended)

Levels: stop = tighter of 1.5×ATR and the 10-day low, but at least 1×ATR away;
T1 = 1.5R (book half, stop to cost), T2 = 3R; after T1 a close below EMA20 exits;
time stop after 15 trading days.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_signals.models import Candidate, Position
from stock_signals.regime import Regime


@dataclass
class SwingRules:
    universe: set[str]
    version: str = "swing-v1"
    book: str = "swing"
    min_traded_value: float = 10e7  # ₹10 crore median daily value
    atr_stop: float = 1.5
    min_stop_atr: float = 1.0
    t1_r: float = 1.5
    t2_r: float = 3.0
    time_stop_days: int = 15
    breakout_volume: float = 1.5
    max_rsi: float = 80.0
    entry_band_atr: float = 0.3
    base_new_per_day: int = 5

    def max_new(self, regime: Regime) -> int:
        return int(self.base_new_per_day * regime.swing_multiplier)

    def candidates(self, snap: pd.DataFrame, regime: Regime, week_end: bool) -> list[Candidate]:
        s = snap[snap.index.isin(self.universe)]
        s = s[
            (s["bars_count"] >= 200)
            & (s["traded_value_med20"] >= self.min_traded_value)
            & (s["close"] > s["sma50"])
            & (s["sma50"] > s["sma200"])
            & (s["rsi14"] < self.max_rsi)
            & s["atr14"].gt(0)
        ]
        if s.empty:
            return []
        breakout = (s["close"] > s["high20_prev"]) & (s["vol_ratio20"] >= self.breakout_volume)
        pullback = (
            s["touched_ema20_3d"].astype(bool)
            & (s["close"] > s["ema20"])
            & (s["close"] > s["prev_close"])
            & ((s["close"] - s["ema20"]) <= s["atr14"])
        )
        s = s[breakout | pullback].assign(
            setup=np.where(breakout[breakout | pullback], "breakout", "pullback")
        )
        out = []
        for symbol, r in s.sort_values("roc126", ascending=False).iterrows():
            close, atr = float(r["close"]), float(r["atr14"])
            stop = max(close - self.atr_stop * atr, float(r["low10"]))
            stop = min(stop, close - self.min_stop_atr * atr)
            if stop <= 0:
                continue
            risk = close - stop
            if r["setup"] == "breakout":
                reason = f"20d breakout, vol {r['vol_ratio20']:.1f}x"
            else:
                reason = "pullback to 20 EMA, bounce"
            reason += f", 6m RS {r['roc126']:+.0%}, RSI {r['rsi14']:.0f}"
            out.append(
                Candidate(
                    symbol=str(symbol),
                    ref_price=round(close, 2),
                    entry_low=round(close - 0.5 * atr, 2),
                    entry_high=round(close + self.entry_band_atr * atr, 2),
                    stop=round(stop, 2),
                    t1=round(close + self.t1_r * risk, 2),
                    t2=round(close + self.t2_r * risk, 2),
                    time_stop_days=self.time_stop_days,
                    score=float(r["roc126"]),
                    reason=reason,
                )
            )
        return out

    def close_exit(self, pos: Position, row: pd.Series, week_end: bool) -> str | None:
        if pos.time_stop_days and pos.bars_held >= pos.time_stop_days:
            return "time_stop"
        if pos.t1_done and pd.notna(row.get("ema20")) and row["close"] < row["ema20"]:
            return "trail_ema20"
        return None
