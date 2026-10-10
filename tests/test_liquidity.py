import pandas as pd

from stock_signals.liquidity import median_traded_value
from tests.conftest import make_bars, weekdays


def test_median_uses_latest_window():
    days = weekdays("2026-01-01", "2026-03-31")
    old = make_bars("A", days[:-20], start_price=100, volume=1)
    new = make_bars("A", days[-20:], start_price=10, volume=100)
    new["close"] = 10.0
    out = median_traded_value(pd.concat([old, new]))
    assert out == {"A": 1000.0}
