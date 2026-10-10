"""Model features. Every column on day D uses data up to D's close only.

Works on the full panel (many dates) or a one-date snapshot: cross-sectional features
(sector momentum, ranks) are computed per date either way, so training and live
scoring see identical inputs.
"""

import numpy as np
import pandas as pd

BASE = [
    "rsi14",
    "roc63",
    "roc126",
    "roc252",
    "vol252",
    "vol_ratio20",
    "mom_score",
]
DERIVED = [
    "atr_pct",
    "dist_high20",
    "dist_sma20",
    "dist_sma50",
    "dist_sma200",
    "sma50_over_200",
    "ret5",
    "liquidity_log",
    "sector_mom63",
    "rs_vs_sector63",
    "roc126_rank",
    "nifty_vs_sma200",
    "breadth50",
]
KRONOS = ["kronos_p_up", "kronos_exp_ret", "kronos_spread", "kronos_p_t1"]
FEATURES = BASE + DERIVED


def make_features(
    df: pd.DataFrame,
    sector_of: dict[str, str],
    regimes: pd.DataFrame,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`df` has `symbol`, `date` and indicator columns. `extra` (optional) is indexed by
    (date, symbol) with additional feature columns, e.g. Kronos outputs."""
    out = pd.DataFrame(index=df.index)
    out["date"], out["symbol"] = df["date"].values, df["symbol"].values
    for c in BASE:
        out[c] = df[c].astype(float)
    close = df["close"].astype(float)
    out["atr_pct"] = df["atr14"] / close
    out["dist_high20"] = close / df["high20_prev"] - 1
    out["dist_sma20"] = close / df["sma20"] - 1
    out["dist_sma50"] = close / df["sma50"] - 1
    out["dist_sma200"] = close / df["sma200"] - 1
    out["sma50_over_200"] = df["sma50"] / df["sma200"] - 1
    out["ret5"] = close / df["close_5"] - 1
    out["liquidity_log"] = np.log10(df["traded_value_med20"].clip(lower=1))
    sector = df["symbol"].map(sector_of).fillna("")
    out["sector_mom63"] = df.groupby([df["date"], sector])["roc63"].transform("mean")
    out["rs_vs_sector63"] = df["roc63"] - out["sector_mom63"]
    out["roc126_rank"] = df.groupby("date")["roc126"].rank(pct=True)
    reg = regimes.reindex(df["date"].values)
    out["nifty_vs_sma200"] = (reg["nifty_close"] / reg["nifty_sma200"] - 1).values
    out["breadth50"] = reg["breadth50"].values
    if extra is not None and not extra.empty:
        keyed = out.set_index(["date", "symbol"]).index
        for c in extra.columns:
            out[c] = extra[c].reindex(keyed).values
    return out
