import pandas as pd

WINDOW = 20


def median_traded_value(bars: pd.DataFrame, window: int = WINDOW) -> dict[str, float]:
    """20-day median of close × volume (₹) per symbol, using the latest `window` bars."""
    if bars.empty:
        return {}
    b = bars.sort_values(["symbol", "date"]).groupby("symbol", group_keys=False).tail(window)
    value = b["close"] * b["volume"]
    return value.groupby(b["symbol"]).median().round(0).to_dict()
