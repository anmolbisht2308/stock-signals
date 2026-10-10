import numpy as np
import pandas as pd
import pytest

from stock_signals.indicators import add_indicators, add_momentum
from tests.conftest import make_bars, weekdays
from tests.factories import calendar, series, trend

DAYS = weekdays("2025-01-01", "2026-06-30")


def test_basic_indicators_on_linear_series():
    df = add_indicators(make_bars("A", DAYS[:260]))
    last = df.iloc[-1]
    closes = df["close"]
    assert last["sma20"] == pytest.approx(closes.tail(20).mean())
    assert last["sma200"] == pytest.approx(closes.tail(200).mean())
    assert last["atr14"] == pytest.approx(4.0)  # constant true range
    assert last["rsi14"] == 100.0  # only up days
    assert last["high20_prev"] == pytest.approx(df["high"].iloc[-21:-1].max())
    assert last["roc63"] == pytest.approx(closes.iloc[-1] / closes.iloc[-64] - 1)
    assert last["bars_count"] == 260
    assert np.isnan(df.iloc[198]["sma200"]) and not np.isnan(df.iloc[199]["sma200"])


def test_no_look_ahead():
    closes = trend(300, wiggle=0.03)
    days = calendar(300)
    full = add_indicators(series("A", closes, days))
    part = add_indicators(series("A", closes[:250], days))
    cols = [c for c in full.columns if c not in ("symbol", "date")]
    pd.testing.assert_frame_equal(
        full.iloc[:250][cols].reset_index(drop=True), part[cols].reset_index(drop=True)
    )


def test_groups_do_not_leak_between_symbols():
    days = calendar(60)
    a = series("A", trend(60), days)
    b = series("B", trend(60, start=500), days)
    both = add_indicators(pd.concat([a, b]))
    alone = add_indicators(b)
    assert both[both.symbol == "B"]["sma20"].tolist() == pytest.approx(
        alone["sma20"].tolist(), nan_ok=True
    )


def test_volume_ratio_uses_previous_days():
    days = calendar(30)
    vol = [1e6] * 29 + [3e6]
    df = add_indicators(series("A", trend(30), days, volume=vol))
    assert df.iloc[-1]["vol_ratio20"] == pytest.approx(3.0)


def test_momentum_rank_per_date():
    days = calendar(300)
    strong = series("S", trend(300, daily=0.003, wiggle=0.005), days)
    weak = series("W", trend(300, daily=0.0005, wiggle=0.005), days)
    illiquid = series("I", trend(300, daily=0.004, wiggle=0.005), days, volume=10)
    df = add_momentum(add_indicators(pd.concat([strong, weak, illiquid])), min_traded_value=1e6)
    last = df[df.date == days[-1]].set_index("symbol")
    assert last.loc["S", "mom_rank"] == 1 and last.loc["W", "mom_rank"] == 2
    assert np.isnan(last.loc["I", "mom_rank"])
    assert df[df.date == days[100]]["mom_rank"].isna().all()  # < 252 bars
