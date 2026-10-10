import pandas as pd

from stock_signals.indicators import add_indicators
from stock_signals.panel import week_end_dates
from stock_signals.regime import regime_on, regime_table
from tests.factories import calendar, series, trend


def panel(nifty_daily, stock_daily):
    days = calendar(260)
    frames = [series("NIFTY50", trend(260, daily=nifty_daily, wiggle=0.002), days)]
    frames += [
        series(f"S{i}", trend(260, daily=d, wiggle=0.002), days) for i, d in enumerate(stock_daily)
    ]
    return add_indicators(pd.concat(frames)), days


def test_modes():
    p, days = panel(0.001, [0.002, 0.002, -0.002])
    t = regime_table(p, "NIFTY50", {"S0", "S1", "S2"})
    r = regime_on(t, days[-1])
    assert r.mode == "risk_on" and abs(r.breadth50 - 2 / 3) < 1e-9
    assert r.swing_multiplier == 1.0
    assert t.loc[days[10], "mode"] == "risk_off"  # no 200-DMA yet

    p, days = panel(0.001, [-0.002, -0.002, 0.002])
    assert regime_on(regime_table(p, "NIFTY50", {"S0", "S1", "S2"}), days[-1]).mode == "caution"

    p, days = panel(-0.001, [0.002])
    r = regime_on(regime_table(p, "NIFTY50", {"S0"}), days[-1])
    assert r.mode == "risk_off" and not r.positional_entries_allowed
    assert "risk-off" in r.describe()


def test_week_end_dates_handles_holidays_and_partial_week():
    cal = [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",  # Fri 9th holiday
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
        "2026-01-16",
        "2026-01-19",
        "2026-01-20",
    ]  # current week, incomplete
    assert week_end_dates(cal) == {"2026-01-08", "2026-01-16"}
    assert week_end_dates(cal[:-2]) == {"2026-01-08", "2026-01-16"}
