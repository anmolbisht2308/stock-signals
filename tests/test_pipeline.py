from datetime import datetime

import pandas as pd
import pytest

from stock_signals.bars import BarStore
from stock_signals.dates import IST
from stock_signals.pipeline import Pipeline, PipelineError
from stock_signals.universe import Member, SymbolStore
from tests.conftest import FakeSource, make_bars, weekdays

DAYS = weekdays("2025-12-01", "2026-01-30")
AFTER_CLOSE = datetime(2026, 1, 30, 17, 0, tzinfo=IST)


def setup(db, data, **kw):
    symbols = SymbolStore(db)
    source = FakeSource(data, **kw)
    p = Pipeline(BarStore(db), symbols, source, history_years=1, batch_size=2)
    p.setup()
    symbols.sync_index(
        "NIFTY500",
        [Member(s, s, "Ind", "") for s in data if s.isalpha() and s not in ("INDIAVIX",)],
    )
    return p, source


def data_until(end, symbols=("NIFTY50", "INDIAVIX", "AAA", "BBB", "CCC")):
    days = [d for d in DAYS if d <= end]
    out = {}
    for s in symbols:
        df = make_bars(s, days)
        if s in ("NIFTY50", "INDIAVIX"):
            df["volume"] = 0
        out[s] = df
    return out


def test_backfill_then_incremental_update(db):
    p, source = setup(db, data_until("2026-01-23"))
    res = p.run(full=True, now=datetime(2026, 1, 23, 17, 0, tzinfo=IST))
    assert res.symbols == 5 and res.failed == []
    assert res.quality.ok, res.quality.summary()
    assert p.bars.last_dates()["AAA"] == "2026-01-23"

    source.data = data_until("2026-01-30")
    source.calls.clear()
    res = p.run(now=AFTER_CLOSE)
    assert res.quality.ok, res.quality.summary()
    # Incremental fetch starts OVERLAP_DAYS before the last stored bar.
    assert {str(c[1]) for c in source.calls} == {"2026-01-16"}
    assert p.bars.last_dates()["CCC"] == "2026-01-30"
    assert db["bars"].count_documents({"symbol": "AAA"}) == len(DAYS)
    assert p.symbols.get("AAA")["median_traded_value_20d"] > 0


def test_partial_bar_before_close_is_not_stored(db):
    p, _ = setup(db, data_until("2026-01-30"))
    p.run(full=True, now=datetime(2026, 1, 30, 11, 0, tzinfo=IST))
    assert p.bars.last_dates()["AAA"] == "2026-01-29"


def test_new_symbol_gets_full_history(db):
    p, source = setup(db, data_until("2026-01-30"))
    p.run(full=True, only=["AAA"], now=AFTER_CLOSE)
    source.calls.clear()
    p.run(only=["AAA", "BBB"], now=AFTER_CLOSE)
    starts = {tuple(c[0]): str(c[1]) for c in source.calls}
    assert starts[("BBB",)] == "2025-01-30"
    assert db["bars"].count_documents({"symbol": "BBB"}) == len(DAYS)


def test_failed_symbols_reported(db):
    p, _ = setup(db, data_until("2026-01-30"), fail={"BBB"})
    res = p.run(full=True, now=AFTER_CLOSE)
    assert res.failed == ["BBB"]
    assert any(i.kind == "fetch_failed" for i in res.quality.issues)
    assert "Failed (1): BBB" in res.summary()


def test_missing_calendar_raises(db):
    p, _ = setup(db, data_until("2026-01-30"), fail={"NIFTY50"})
    with pytest.raises(PipelineError):
        p.run(full=True, now=AFTER_CLOSE)


def test_new_split_triggers_full_refetch(db):
    data = data_until("2026-01-23")
    p, source = setup(db, data)
    p.run(full=True, now=datetime(2026, 1, 23, 17, 0, tzinfo=IST))

    # 2:1 split on 2026-01-28: the source now serves the whole history halved.
    new = data_until("2026-01-30")
    new["AAA"][["open", "high", "low", "close", "adj_close"]] /= 2
    source.data = new
    source.splits = pd.DataFrame([{"symbol": "AAA", "date": "2026-01-28", "ratio": 2.0}])
    res = p.run(now=AFTER_CLOSE)
    assert res.refetched == ["AAA"]
    first = p.bars.load("AAA").iloc[0]
    assert first["close"] == pytest.approx(new["AAA"].iloc[0]["close"])
    assert not any(i.kind == "big_jump" for i in res.quality.issues)
