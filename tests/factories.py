"""Synthetic price series and snapshots for rule tests."""

import numpy as np
import pandas as pd

from stock_signals.sources import BAR_COLUMNS
from tests.conftest import weekdays


def series(
    symbol: str,
    closes: list[float] | np.ndarray,
    days: list[str],
    volume: float | list[float] = 1e6,
    spread: float = 0.01,
) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    vol = np.full(len(closes), volume, dtype=float) if np.isscalar(volume) else np.asarray(volume)
    opens = np.r_[closes[0], closes[:-1]]
    high = np.maximum(opens, closes) * (1 + spread)
    low = np.minimum(opens, closes) * (1 - spread)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": days[: len(closes)],
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "adj_close": closes,
            "volume": vol,
        }
    )[BAR_COLUMNS]


def trend(
    n: int, start: float = 100.0, daily: float = 0.002, wiggle: float = 0.01, phase: float = 0.0
) -> np.ndarray:
    t = np.arange(n)
    return start * (1 + daily) ** t * (1 + wiggle * np.sin(t / 3 + phase))


def calendar(n: int = 320) -> list[str]:
    return weekdays("2025-01-01", "2026-12-31")[:n]


SNAP_DEFAULTS = dict(
    bars_count=300,
    traded_value_med20=5e8,
    close=110.0,
    open=108.0,
    prev_close=108.0,
    sma20=105.0,
    sma50=100.0,
    sma200=90.0,
    ema20=106.0,
    atr14=2.0,
    rsi14=60.0,
    high20_prev=112.0,
    low10=104.0,
    vol_ratio20=1.0,
    touched_ema20_3d=False,
    roc63=0.1,
    roc126=0.2,
    roc252=0.4,
    vol252=0.25,
    mom_score=1.2,
    mom_rank=50.0,
)


def snapshot(rows: dict[str, dict]) -> pd.DataFrame:
    data = {sym: {**SNAP_DEFAULTS, **over} for sym, over in rows.items()}
    return pd.DataFrame.from_dict(data, orient="index")
