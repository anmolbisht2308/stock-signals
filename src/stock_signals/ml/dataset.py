"""Training rows and labels.

Swing: one row per rule candidate (date, symbol). Label = 1 if, entering at the next
open, T1 was reached before the stop within the time stop; 0 otherwise. Candidates that
would have been skipped at the open (gap above the entry zone or below the stop) are
dropped. Positional: one row per eligible stock on each week-end. Label = its forward
60-session return rank within that date (0–1).
"""

import numpy as np
import pandas as pd

from stock_signals.ml.features import make_features
from stock_signals.panel import Panel
from stock_signals.regime import regime_on
from stock_signals.strategies import Strategy

POSITIONAL_HORIZON = 60


def _symbol_arrays(panel: Panel) -> dict[str, tuple[dict[str, int], np.ndarray]]:
    out = {}
    for sym, g in panel.data.groupby("symbol", sort=False):
        g = g.sort_values("date")
        idx = {d: i for i, d in enumerate(g["date"])}
        out[str(sym)] = (idx, g[["open", "high", "low", "close"]].to_numpy(dtype=float))
    return out


def label_t1_before_stop(
    ohlc: np.ndarray, i: int, entry_high: float, stop: float, t1: float, horizon: int
) -> float:
    """NaN when the trade would not be entered or the future is not known yet."""
    if i + 1 >= len(ohlc):
        return np.nan
    open_ = ohlc[i + 1, 0]
    if open_ > entry_high or open_ <= stop:
        return np.nan
    end = min(i + 1 + horizon, len(ohlc))
    for j in range(i + 1, end):
        _, high, low, _ = ohlc[j]
        if low <= stop:
            return 0.0
        if high >= t1:
            return 1.0
    return 0.0 if end - (i + 1) >= horizon else np.nan


def swing_dataset(
    panel: Panel,
    strategy: Strategy,
    sector_of: dict[str, str],
    start: str,
    end: str,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Features + `label` for every swing rule candidate between start and end."""
    arrays = _symbol_arrays(panel)
    feats = make_features(panel.data, sector_of, panel.regimes, extra)
    feats.index = pd.MultiIndex.from_arrays([feats["date"], feats["symbol"]])
    rows = []
    for d, snap in panel.by_date().items():
        if not (start <= d <= end) or d not in panel.regimes.index:
            continue
        regime = regime_on(panel.regimes, d)
        for c in strategy.candidates(snap, regime, d in panel.week_ends):
            idx, ohlc = arrays[c.symbol]
            label = label_t1_before_stop(
                ohlc, idx[d], c.entry_high, c.stop, c.t1 or np.inf, c.time_stop_days or 15
            )
            rows.append((d, c.symbol, label))
    if not rows:
        return pd.DataFrame()
    keys = pd.MultiIndex.from_tuples([(d, s) for d, s, _ in rows])
    out = feats.loc[keys].reset_index(drop=True)
    out["label"] = [lab for _, _, lab in rows]
    return out


def positional_dataset(
    panel: Panel,
    sector_of: dict[str, str],
    start: str,
    end: str,
    extra: pd.DataFrame | None = None,
    horizon: int = POSITIONAL_HORIZON,
) -> pd.DataFrame:
    data = panel.data
    fwd = data.groupby("symbol", sort=False)["close"].shift(-horizon) / data["close"] - 1
    mask = (
        data["date"].isin(panel.week_ends)
        & data["mom_score"].notna()
        & (data["date"] >= start)
        & (data["date"] <= end)
    )
    feats = make_features(data, sector_of, panel.regimes, extra)[mask]
    label = fwd[mask].groupby(data.loc[mask, "date"]).rank(pct=True)
    feats["label"] = label.values
    return feats.reset_index(drop=True)
