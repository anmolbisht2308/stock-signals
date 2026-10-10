import pandas as pd
import pytest

from stock_signals.models import Position
from stock_signals.regime import Regime
from stock_signals.strategies.positional import PositionalRules
from stock_signals.strategies.swing import SwingRules
from tests.factories import snapshot

ON = Regime("2026-01-09", "risk_on", 25000, 23000, 0.6)
OFF = Regime("2026-01-09", "risk_off", 22000, 23000, 0.3)


def swing(universe=("BRK", "PUL", "FLAT", "ILL", "HOT", "OUT")):
    return SwingRules(universe=set(universe))


def test_swing_breakout_levels():
    snap = snapshot({"BRK": dict(close=115.0, high20_prev=112.0, vol_ratio20=2.0)})
    [c] = swing().candidates(snap, ON, False)
    # stop = max(115 - 1.5*2, low10 104) = 112, at least 1 ATR away -> 112 ; R = 3
    assert (c.symbol, c.stop, c.t1, c.t2) == ("BRK", 112.0, 119.5, 124.0)
    assert c.entry_high == pytest.approx(115.6) and c.time_stop_days == 15
    assert "breakout" in c.reason


def test_swing_stop_respects_min_distance():
    snap = snapshot({"BRK": dict(close=115.0, low10=114.5, vol_ratio20=2.0)})
    [c] = swing().candidates(snap, ON, False)
    assert c.stop == 113.0  # low10 too close -> 1 ATR


def test_swing_pullback_and_filters():
    snap = snapshot(
        {
            "PUL": dict(close=107.0, ema20=106.0, prev_close=106.5, touched_ema20_3d=True),
            "FLAT": dict(close=107.0, sma50=110.0),  # trend broken
            "ILL": dict(close=115.0, vol_ratio20=2.0, traded_value_med20=1e6),  # illiquid
            "HOT": dict(close=115.0, vol_ratio20=2.0, rsi14=85.0),  # overextended
            "OUT": dict(close=115.0, vol_ratio20=2.0),  # not in universe
            "NOV": dict(close=115.0, vol_ratio20=1.2),  # weak volume
        }
    )
    got = swing(universe=("PUL", "FLAT", "ILL", "HOT", "NOV")).candidates(snap, ON, False)
    assert [c.symbol for c in got] == ["PUL"] and "pullback" in got[0].reason


def test_swing_sorted_by_relative_strength_and_regime_slots():
    snap = snapshot(
        {
            "A": dict(close=115.0, vol_ratio20=2.0, roc126=0.1),
            "B": dict(close=115.0, vol_ratio20=2.0, roc126=0.5),
        }
    )
    assert [c.symbol for c in swing(("A", "B")).candidates(snap, ON, False)] == ["B", "A"]
    assert swing().max_new(ON) == 5 and swing().max_new(OFF) == 2


def pos(**kw):
    base = dict(
        id="p",
        book="swing",
        symbol="X",
        signal_id="s",
        rule_version="v",
        entry_date="2026-01-01",
        entry_price=100,
        initial_qty=10,
        qty=10,
        stop=95,
        initial_stop=95,
        time_stop_days=15,
    )
    return Position(**{**base, **kw})


def test_swing_close_exits():
    r = SwingRules(universe=set())
    row = pd.Series({"close": 105.0, "ema20": 104.0})
    assert r.close_exit(pos(bars_held=3), row, False) is None
    assert r.close_exit(pos(bars_held=15), row, False) == "time_stop"
    assert (
        r.close_exit(pos(t1_done=True), pd.Series({"close": 103.0, "ema20": 104.0}), False)
        == "trail_ema20"
    )
    assert r.close_exit(pos(), pd.Series({"close": 103.0, "ema20": 104.0}), False) is None


def test_positional_entries_weekly_only():
    rules = PositionalRules(universe={"M1", "M2", "M3", "LOW"})
    snap = snapshot(
        {
            "M1": dict(mom_rank=2.0, mom_score=2.0),
            "M2": dict(mom_rank=1.0, mom_score=3.0),
            "M3": dict(mom_rank=5.0, close=85.0),  # below SMA200
            "LOW": dict(mom_rank=40.0),
        }
    )
    assert rules.candidates(snap, ON, week_end=False) == []
    assert rules.candidates(snap, OFF, week_end=True) == []
    got = rules.candidates(snap, ON, week_end=True)
    assert [c.symbol for c in got] == ["M2", "M1"]
    assert got[0].stop == pytest.approx(104.0) and got[0].t1 is None
    assert rules.max_new(OFF) == 0


def test_positional_close_exits():
    rules = PositionalRules(universe=set())
    p = pos(book="positional", time_stop_days=None)
    ok = pd.Series({"close": 110.0, "sma50": 100.0, "mom_rank": 30.0})
    assert rules.close_exit(p, ok, week_end=False) is None
    assert rules.close_exit(p, ok, week_end=True) is None
    below = pd.Series({"close": 95.0, "sma50": 100.0, "mom_rank": 30.0})
    assert rules.close_exit(p, below, True) == "trail_sma50"
    assert (
        rules.close_exit(p, pd.Series({"close": 110.0, "sma50": 100.0, "mom_rank": 150.0}), True)
        == "rank_exit"
    )
    assert (
        rules.close_exit(
            p, pd.Series({"close": 110.0, "sma50": 100.0, "mom_rank": float("nan")}), True
        )
        == "rank_exit"
    )
