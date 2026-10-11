import numpy as np
import pandas as pd

from stock_signals.panel import Universe, build_panel
from stock_signals.scan import build_strategies, scan, scan_text
from stock_signals.sizing import SizingConfig
from tests.factories import calendar, series, trend

N = 300


def bars():
    days = calendar(N)
    nifty = series("NIFTY50", trend(N, daily=0.0008, wiggle=0.002), days, volume=0)
    vix = series("INDIAVIX", np.full(N, 14.0), days, volume=0)
    # Uptrend with a 20-day breakout on the last day on 3x volume.
    closes = trend(N, daily=0.0015, wiggle=0.02)
    closes[-1] = closes[-25:-1].max() * 1.02
    vol = np.full(N, 2e6)
    vol[-1] = 6e6
    brk = series("BRK", closes, days, volume=vol)
    slow = series("SLOW", trend(N, daily=-0.001, wiggle=0.004), days, volume=2e6)
    return pd.concat([nifty, vix, brk, slow]), days


def test_scan_end_to_end_is_deterministic():
    b, days = bars()
    universe = Universe(
        swing={"BRK", "SLOW"}, positional={"BRK", "SLOW"}, sector_of={"BRK": "Auto", "SLOW": "IT"}
    )
    panel = build_panel(b, universe)
    args = (
        panel,
        universe,
        days[-1],
        build_strategies(universe),
        {"swing": 200_000, "positional": 300_000},
        SizingConfig(),
    )
    first, second = scan(*args), scan(*args)
    assert first.regime.mode == "risk_on"
    swing = first.signals["swing"]
    assert [s.symbol for s in swing] == ["BRK"]
    assert swing[0].stop < swing[0].ref_price < swing[0].t1 < swing[0].t2
    assert first.signals == second.signals
    text = scan_text(first)
    assert "SWING BUY BRK" in text and "Market risk-on" in text


def test_build_strategies_without_swing():
    universe = Universe(swing={"A"}, positional={"A"}, sector_of={})
    assert [s.book for s in build_strategies(universe)] == ["swing", "positional"]
    assert [s.book for s in build_strategies(universe, swing=False)] == ["positional"]
