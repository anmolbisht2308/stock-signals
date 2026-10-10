from stock_signals.models import Candidate
from stock_signals.sizing import SizingConfig, position_qty, select_signals

CFG = SizingConfig(risk_pct=1.0, max_position_pct=20.0, max_per_sector=1, max_positions=3)


def test_qty_by_risk_and_cap():
    assert position_qty(100_000, 100, 90, CFG) == 100  # ₹1,000 risk / ₹10
    assert position_qty(100_000, 100, 98, CFG) == 200  # risk allows 500, 20% cap allows 200
    assert position_qty(100_000, 100, 101, CFG) == 0


def cand(symbol, price=100.0, stop=90.0):
    return Candidate(
        symbol=symbol,
        ref_price=price,
        entry_low=price - 1,
        entry_high=price + 1,
        stop=stop,
        reason="r",
    )


def run(cands, **kw):
    args = dict(
        book="swing",
        date="2026-01-09",
        version="swing-v1",
        equity=100_000,
        free_cash=100_000,
        held=set(),
        sector_of={"A": "IT", "B": "IT", "C": "Bank", "D": "Auto", "E": "Pharma"},
        sector_counts={},
        open_count=0,
        max_new=5,
        cfg=CFG,
    )
    return [s.symbol for s in select_signals(cands, **{**args, **kw})]


def test_limits():
    cands = [cand(s) for s in "ABCDE"]
    assert run(cands) == ["A", "C", "D"]  # sector cap 1, max 3 positions
    assert run(cands, held={"A"}) == ["B", "C", "D"]
    assert run(cands, open_count=2) == ["A"]
    assert run(cands, max_new=1) == ["A"]
    assert run(cands, sector_counts={"IT": 1}) == ["C", "D", "E"]
    assert run(cands, free_cash=15_000) == ["A"]  # each costs 100 × ₹101


def test_signal_fields():
    [s] = select_signals(
        [cand("A")],
        book="swing",
        date="2026-01-09",
        version="swing-v1",
        equity=100_000,
        free_cash=100_000,
        held=set(),
        sector_of={"A": "IT"},
        sector_counts={},
        open_count=0,
        max_new=5,
        cfg=CFG,
    )
    assert s.id == "swing:swing-v1:A:2026-01-09"
    assert (s.qty, s.risk_per_share, s.risk_amount, s.sector) == (100, 10.0, 1000.0, "IT")
