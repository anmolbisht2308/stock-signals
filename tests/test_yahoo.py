from datetime import date

import pandas as pd

from stock_signals.sources.yahoo import YahooSource, normalise, to_ticker


def yf_frame(tickers, days, split_on=None):
    idx = pd.DatetimeIndex(days)
    cols = {}
    for t in tickers:
        n = len(days)
        base = pd.Series(range(n), index=idx, dtype=float) + 100
        fields = {
            "Open": base,
            "High": base + 2,
            "Low": base - 2,
            "Close": base + 1,
            "Adj Close": base + 0.5,
            "Volume": pd.Series(1000.0, index=idx),
            "Dividends": pd.Series(0.0, index=idx),
            "Stock Splits": pd.Series(0.0, index=idx),
        }
        if split_on and t in split_on:
            fields["Stock Splits"].loc[split_on[t]] = 2.0
        for f, s in fields.items():
            cols[(t, f)] = s
    return pd.DataFrame(cols)


def test_ticker_mapping():
    assert to_ticker("RELIANCE") == "RELIANCE.NS"
    assert to_ticker("M&M") == "M&M.NS"
    assert to_ticker("NIFTY50") == "^NSEI"


def test_normalise_converts_columns_and_splits():
    days = ["2026-01-05", "2026-01-06", "2026-01-07"]
    raw = yf_frame(["TCS.NS"], days, split_on={"TCS.NS": "2026-01-06"})
    bars, splits = normalise(raw["TCS.NS"], "TCS")
    assert list(bars["date"]) == days
    assert bars.iloc[0][["open", "close", "adj_close", "volume"]].tolist() == [
        100,
        101,
        100.5,
        1000,
    ]
    assert splits.to_dict("records") == [{"symbol": "TCS", "date": "2026-01-06", "ratio": 2.0}]


def test_normalise_drops_empty_rows_and_tz_converts():
    idx = pd.DatetimeIndex(["2026-01-05 03:45", "2026-01-06 03:45"], tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [1.0, None],
            "High": [2.0, None],
            "Low": [0.5, None],
            "Close": [1.5, None],
            "Volume": [10.0, None],
        },
        index=idx,
    )
    bars, _ = normalise(df, "X")
    assert bars["date"].tolist() == ["2026-01-05"]


def test_fetch_marks_missing_tickers_failed_and_end_is_inclusive():
    days = ["2026-01-05", "2026-01-06"]
    calls = {}

    def download(**kw):
        calls.update(kw)
        raw = yf_frame(["INFY.NS", "BAD.NS"], days)
        for c in [c for c in raw.columns if c[0] == "BAD.NS"]:
            raw[c] = float("nan")
        return raw

    res = YahooSource(download=download).fetch(["INFY", "BAD"], date(2026, 1, 5), date(2026, 1, 6))
    assert calls["end"] == "2026-01-07"
    assert calls["auto_adjust"] is False
    assert res.failed == ["BAD"]
    assert res.bars["symbol"].unique().tolist() == ["INFY"]


def test_fetch_retries_then_succeeds():
    attempts = []

    def download(**kw):
        attempts.append(1)
        if len(attempts) < 2:
            raise ConnectionError("boom")
        return yf_frame(["INFY.NS"], ["2026-01-05"])

    res = YahooSource(download=download, pause=0).fetch(
        ["INFY"], date(2026, 1, 5), date(2026, 1, 5)
    )
    assert len(attempts) == 2 and len(res.bars) == 1
