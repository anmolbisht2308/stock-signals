import pandas as pd
import pytest

from stock_signals.book import BookEngine, BookState
from stock_signals.costs import CostModel
from stock_signals.models import Candidate
from stock_signals.regime import Regime
from stock_signals.sizing import SizingConfig
from stock_signals.strategies.swing import SwingRules

ON = Regime("d", "risk_on", 1, 0, 0.6)
NO_COSTS = CostModel(
    stt_pct=0,
    exchange_pct=0,
    sebi_per_crore=0,
    stamp_buy_pct=0,
    gst_pct=0,
    dp_per_sell=0,
    slippage_pct=0,
)


class Stub:
    """Swing exits, but entries come from `self.next`."""

    book = "swing"
    version = "stub-v1"

    def __init__(self):
        self.next: list[Candidate] = []
        self.rules = SwingRules(universe=set())

    def candidates(self, snap, regime, week_end):
        out, self.next = self.next, []
        return out

    def max_new(self, regime):
        return 5

    def close_exit(self, pos, row, week_end):
        return self.rules.close_exit(pos, row, week_end)


def bar(o, h, l, c, ema20=0.0):  # noqa: E741
    return {"open": o, "high": h, "low": l, "close": c, "ema20": ema20}


def snap(**bars):
    return pd.DataFrame.from_dict(bars, orient="index")


def cand(symbol="X", ref=100.0, stop=95.0, t1=107.5, t2=115.0, entry_high=101.0, days=15):
    return Candidate(
        symbol=symbol,
        ref_price=ref,
        entry_low=99,
        entry_high=entry_high,
        stop=stop,
        t1=t1,
        t2=t2,
        time_stop_days=days,
        reason="test",
    )


def engine(costs=NO_COSTS, capital=100_000.0):
    strat = Stub()
    eng = BookEngine(
        strat,
        BookState.new("swing", "stub-v1", capital),
        costs,
        SizingConfig(risk_pct=1.0, max_position_pct=100.0),
        {},
    )
    return eng, strat


def run(eng, strat, days):
    """days: list of (snapshot, candidates_for_tomorrow)."""
    out = []
    for i, (s, cands) in enumerate(days):
        strat.next = cands
        out.append(eng.step(f"2026-01-{i + 1:02d}", s, ON, False))
    return out


def kinds(res):
    return [e.kind for e in res.events]


def test_signal_fills_at_next_open_then_stop():
    eng, strat = engine()
    r = run(
        eng,
        strat,
        [
            (snap(X=bar(100, 101, 99, 100)), [cand()]),
            (snap(X=bar(100.5, 102, 99, 101)), []),
            (snap(X=bar(100, 100, 94, 95)), []),
        ],
    )
    assert kinds(r[0]) == ["signal"]
    assert kinds(r[1]) == ["fill"]
    pos_qty = r[0].events[0].data["signal"]["qty"]
    assert pos_qty == 200  # ₹1,000 risk / ₹5
    assert kinds(r[2]) == ["exit"]
    [t] = r[2].trades
    assert t.entry_price == 100.5 and t.avg_exit_price == 95.0
    assert t.net_pnl == pytest.approx(-5.5 * 200) and t.exit_reason == "stop"
    assert eng.state.cash == pytest.approx(100_000 - 1100)


def test_gap_down_exits_at_open():
    eng, strat = engine()
    r = run(
        eng,
        strat,
        [
            (snap(X=bar(100, 101, 99, 100)), [cand()]),
            (snap(X=bar(100, 101, 99, 100)), []),
            (snap(X=bar(90, 92, 89, 91)), []),
        ],
    )
    assert r[2].trades[0].avg_exit_price == 90


def test_skip_when_open_gaps_above_entry_or_below_stop():
    eng, strat = engine()
    r = run(
        eng,
        strat,
        [
            (snap(X=bar(100, 101, 99, 100), Y=bar(100, 101, 99, 100)), [cand("X"), cand("Y")]),
            (snap(X=bar(103, 104, 102, 103), Y=bar(94, 96, 93, 95)), []),
        ],
    )
    assert kinds(r[1]) == ["skip", "skip"]
    assert "above entry" in r[1].events[0].text and "below stop" in r[1].events[1].text
    assert eng.state.positions == []


def test_t1_books_half_moves_stop_then_t2():
    eng, strat = engine()
    r = run(
        eng,
        strat,
        [
            (snap(X=bar(100, 101, 99, 100)), [cand()]),
            (snap(X=bar(100, 108, 99.5, 107)), []),
            (snap(X=bar(110, 116, 109, 115)), []),
        ],
    )
    assert kinds(r[1]) == ["fill", "partial_exit", "stop_moved"]
    assert eng.state.cash > 0
    assert kinds(r[2]) == ["exit"]
    [t] = r[2].trades
    assert t.avg_exit_price == pytest.approx((107.5 + 115) / 2)
    assert t.exit_reason == "t2" and t.r_multiple == pytest.approx(2.25)


def test_stop_wins_when_stop_and_target_in_same_bar():
    eng, strat = engine()
    r = run(
        eng,
        strat,
        [
            (snap(X=bar(100, 101, 99, 100)), [cand()]),
            (snap(X=bar(100, 109, 94, 100)), []),
        ],
    )
    assert kinds(r[1]) == ["fill", "exit"] and r[1].trades[0].exit_reason == "stop"


def test_breakeven_stop_after_t1():
    eng, strat = engine()
    r = run(
        eng,
        strat,
        [
            (snap(X=bar(100, 101, 99, 100)), [cand()]),
            (snap(X=bar(100, 108, 99.5, 107)), []),
            (snap(X=bar(101, 102, 99, 99.5)), []),
        ],
    )
    [t] = r[2].trades
    assert t.exit_reason == "breakeven_stop" and t.net_pnl == pytest.approx(7.5 * 100)


def test_time_stop_exits_next_open_and_missing_bar_waits():
    eng, strat = engine()
    days = [(snap(X=bar(100, 101, 99, 100)), [cand(days=2)])]
    days += [(snap(X=bar(100, 101, 99, 100)), []), (snap(X=bar(100, 101, 99, 100)), [])]
    days += [(snap(Y=bar(1, 1, 1, 1)), []), (snap(X=bar(102, 103, 101, 102)), [])]
    r = run(eng, strat, days)
    assert kinds(r[2]) == ["exit_planned"]
    assert kinds(r[3]) == []  # no bar for X today
    assert r[4].trades[0].exit_reason == "time_stop" and r[4].trades[0].avg_exit_price == 102


def test_trailing_ema_after_t1():
    eng, strat = engine()
    r = run(
        eng,
        strat,
        [
            (snap(X=bar(100, 101, 99, 100)), [cand()]),
            (snap(X=bar(100, 108, 99.5, 107, ema20=104)), []),
            (snap(X=bar(106, 107, 102, 103, ema20=104)), []),
            (snap(X=bar(103, 104, 102, 103)), []),
        ],
    )
    assert kinds(r[2]) == ["exit_planned"]
    assert r[3].trades[0].exit_reason == "trail_ema20"


def test_costs_flow_into_cash_and_trade():
    costs = CostModel()
    eng, strat = engine(costs=costs)
    r = run(
        eng,
        strat,
        [
            (snap(X=bar(100, 101, 99, 100)), [cand()]),
            (snap(X=bar(100, 101, 99, 100)), []),
            (snap(X=bar(100, 100, 94, 95)), []),
        ],
    )
    [t] = r[2].trades
    assert t.costs > 0 and t.net_pnl == pytest.approx(t.gross_pnl - t.costs, abs=0.02)
    assert eng.state.cash == pytest.approx(100_000 + t.net_pnl, abs=0.02)


def test_equity_marks_open_positions():
    eng, strat = engine()
    r = run(
        eng,
        strat,
        [
            (snap(X=bar(100, 101, 99, 100)), [cand()]),
            (snap(X=bar(100, 103, 99, 102)), []),
        ],
    )
    assert r[1].equity == pytest.approx(100_000 + 2 * 200)
    assert r[1].invested == pytest.approx(102 * 200)


def test_state_round_trips_through_json():
    eng, strat = engine()
    run(
        eng,
        strat,
        [(snap(X=bar(100, 101, 99, 100)), [cand()]), (snap(X=bar(100, 103, 99, 102)), [cand("Y")])],
    )
    again = BookState.model_validate_json(eng.state.model_dump_json())
    assert again == eng.state and again.positions[0].symbol == "X"
