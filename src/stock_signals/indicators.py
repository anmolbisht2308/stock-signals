"""Technical indicators for a long panel of daily bars (one row per symbol and date).

Every value on day D uses data up to D's close only. Columns that describe
"before today" (prev_*, high20_prev, …) are shifted so a breakout compares
today's close against the previous N days.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _by_symbol(df: pd.DataFrame, col: str):
    return df.groupby("symbol", sort=False)[col]


def _wilder(series: pd.Series, symbols: pd.Series, n: int) -> pd.Series:
    return series.groupby(symbols, sort=False).transform(
        lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    )


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.sort_values(["symbol", "date"]).reset_index(drop=True)
    sym = df["symbol"]
    close = _by_symbol(df, "close")

    df["bars_count"] = df.groupby("symbol", sort=False).cumcount() + 1
    df["prev_close"] = close.shift(1)
    df["prev_high"] = _by_symbol(df, "high").shift(1)

    for n in (20, 50, 200):
        df[f"sma{n}"] = close.transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
    df["ema20"] = close.transform(lambda s: s.ewm(span=20, adjust=False, min_periods=20).mean())

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = _wilder(tr, sym, 14)

    change = df["close"] - df["prev_close"]
    gain = _wilder(change.clip(lower=0), sym, 14)
    loss = _wilder((-change).clip(lower=0), sym, 14)
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).where(loss > 0, 100.0).where(gain.notna())

    high = _by_symbol(df, "high")
    df["high20_prev"] = high.transform(lambda s: s.shift(1).rolling(20, min_periods=20).max())
    df["high55_prev"] = high.transform(lambda s: s.shift(1).rolling(55, min_periods=55).max())
    df["low10"] = _by_symbol(df, "low").transform(lambda s: s.rolling(10, min_periods=10).min())

    for n in (63, 126, 252):
        df[f"roc{n}"] = close.transform(lambda s, n=n: s / s.shift(n) - 1)

    ret = close.pct_change()
    df["vol252"] = ret.groupby(sym, sort=False).transform(
        lambda s: s.rolling(TRADING_DAYS, min_periods=200).std() * np.sqrt(TRADING_DAYS)
    )

    vol = _by_symbol(df, "volume")
    avg_vol = vol.transform(lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    df["vol_ratio20"] = df["volume"] / avg_vol.replace(0, np.nan)

    traded = df["close"] * df["volume"]
    df["traded_value_med20"] = traded.groupby(sym, sort=False).transform(
        lambda s: s.rolling(20, min_periods=20).median()
    )

    touched = (df["low"] <= df["ema20"]).astype(float)
    df["touched_ema20_3d"] = (
        touched.groupby(sym, sort=False).transform(lambda s: s.rolling(3, min_periods=1).max()) > 0
    )
    return df


def add_momentum(
    panel: pd.DataFrame, eligible: pd.Series | None = None, min_traded_value: float = 0.0
) -> pd.DataFrame:
    """Momentum score (50% 6-month + 50% 12-month return, divided by volatility) and its
    rank per date among eligible rows (1 = strongest)."""
    df = panel
    score = (0.5 * df["roc126"] + 0.5 * df["roc252"]) / df["vol252"].replace(0, np.nan)
    ok = (
        score.notna()
        & (df["bars_count"] >= TRADING_DAYS)
        & (df["traded_value_med20"].fillna(0) >= min_traded_value)
    )
    if eligible is not None:
        ok &= eligible
    df["mom_score"] = score.where(ok)
    df["mom_rank"] = df.groupby("date")["mom_score"].rank(ascending=False, method="first")
    return df
