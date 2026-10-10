from stock_signals.universe import Member, SymbolStore, read_universe

CSV = """﻿Company Name,Industry,Symbol,Series,ISIN Code
Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018
Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029
Some Bond,Financial Services,XYZ,BE,INE000000000
"""


def test_read_universe_keeps_eq_series(tmp_path):
    p = tmp_path / "n.csv"
    p.write_text(CSV, encoding="utf-8")
    members = read_universe(p)
    assert [m.symbol for m in members] == ["RELIANCE", "TCS"]
    assert members[1].industry == "Information Technology"


def test_sync_index_adds_and_removes(db):
    store = SymbolStore(db)
    a, b, c = (Member(s, s, "Ind", "") for s in ("A", "B", "C"))
    assert store.sync_index("NIFTY200", [a, b]) == {"members": 2, "added": 2, "removed": 0}
    store.sync_index("NIFTY500", [a, b, c])
    assert store.sync_index("NIFTY200", [a]) == {"members": 1, "added": 0, "removed": 1}
    assert store.get("B")["indices"] == ["NIFTY500"] and store.get("B")["status"] == "active"
    store.sync_index("NIFTY500", [a, c])
    assert store.get("B")["status"] == "removed"
    assert store.active(kind="equity") == ["A", "C"]
    assert store.active(index="NIFTY200") == ["A"]


def test_rename_moves_membership(db):
    store = SymbolStore(db)
    store.sync_index("NIFTY500", [Member("OLD", "Co", "Ind", "I1")])
    store.rename("OLD", "NEW")
    assert store.get("OLD")["status"] == "renamed"
    assert store.get("NEW")["indices"] == ["NIFTY500"]
    assert store.active() == ["NEW"]
