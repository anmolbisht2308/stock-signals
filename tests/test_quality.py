import pandas as pd

from stock_signals.quality import check_frame, check_symbol
from tests.conftest import make_bars, weekdays

CAL = weekdays("2026-01-05", "2026-01-30")


def kinds(issues):
    return sorted(i.kind for i in issues)


def test_clean_series_has_no_issues():
    assert check_symbol(make_bars("A", CAL), CAL) == []


def test_missing_days_and_stale():
    bars = make_bars("A", CAL[:10])
    bars = bars[bars["date"] != CAL[3]]
    issues = check_symbol(bars, CAL)
    assert kinds(issues) == ["missing_days", "stale"]


def test_bad_ohlc_zero_volume_duplicates():
    bars = make_bars("A", CAL)
    bars.loc[2, "low"] = bars.loc[2, "close"] + 5
    bars.loc[4, "volume"] = 0
    bars = pd.concat([bars, bars.iloc[[6]]])
    assert kinds(check_symbol(bars, CAL)) == ["bad_ohlc", "duplicate", "zero_volume"]
    assert kinds(check_symbol(bars, CAL, check_volume=False)) == ["bad_ohlc", "duplicate"]


def test_big_jump_unless_split():
    bars = make_bars("A", CAL)
    bars.loc[10:, ["open", "high", "low", "close"]] /= 2
    bars.loc[10:, "high"] += 5
    assert kinds(check_symbol(bars, CAL)) == ["big_jump"]
    assert check_symbol(bars, CAL, split_dates={CAL[10]}) == []


def test_check_frame_summary():
    good, bad = make_bars("A", CAL), make_bars("B", CAL[:5])
    report = check_frame(pd.concat([good, bad]), CAL)
    assert report.checked == 2 and not report.ok
    assert "stale (1)" in report.summary()
