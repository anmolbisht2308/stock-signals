from stock_signals.bars import BarStore
from tests.conftest import make_bars, weekdays


def test_upsert_is_idempotent_and_updates(db):
    store = BarStore(db)
    store.ensure_indexes()
    bars = make_bars("TCS", weekdays("2026-01-05", "2026-01-09"))
    assert store.upsert(bars, "fake") == 5
    bars.loc[0, "close"] = 999.0
    store.upsert(bars, "fake")
    assert db["bars"].count_documents({}) == 5
    assert store.load("TCS").iloc[0]["close"] == 999.0


def test_last_dates_load_and_last_n(db):
    store = BarStore(db)
    store.upsert(make_bars("A", weekdays("2026-01-05", "2026-01-09")), "fake")
    store.upsert(make_bars("B", weekdays("2026-01-05", "2026-01-07")), "fake")
    assert store.last_dates() == {"A": "2026-01-09", "B": "2026-01-07"}
    df = store.load(["A", "B"], start="2026-01-06", last_n=2)
    assert df.groupby("symbol")["date"].apply(list).to_dict() == {
        "A": ["2026-01-08", "2026-01-09"],
        "B": ["2026-01-06", "2026-01-07"],
    }
    assert store.dates("B", start="2026-01-06") == ["2026-01-06", "2026-01-07"]
