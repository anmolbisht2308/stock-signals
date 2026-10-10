import pandas as pd
import pytest

from stock_signals.basket_run import BasketPaper, build_baskets, run_basket_backtest, split_capital
from stock_signals.baskets import (
    BasketBook,
    BasketState,
    CoreRules,
    SatelliteRules,
    cap_weights,
    first_sessions,
)
from stock_signals.costs import CostModel
from stock_signals.panel import Universe, build_panel
from stock_signals.regime import Regime
from stock_signals.views import basket_text
from tests.test_backtest import market

ON = Regime("d", "risk_on", 1, 0, 0.6)
OFF = Regime("d", "risk_off", 1, 2, 0.2)
NO_COSTS = CostModel(
    stt_pct=0,
    exchange_pct=0,
    sebi_per_crore=0,
    stamp_buy_pct=0,
    gst_pct=0,
    dp_per_sell=0,
    slippage_pct=0,
)


def test_cap_weights_stock_and_sector_caps():
    raw = {"A": 10, "B": 5, "C": 1, "D": 1, "E": 1, "F": 1}
    w = cap_weights(raw, cap=0.3)
    assert sum(w.values()) == pytest.approx(1) and max(w.values()) <= 0.3 + 1e-9
    sectors = {"A": "IT", "B": "IT", "C": "Bank", "D": "Auto", "E": "Pharma", "F": "FMCG"}
    w = cap_weights(raw, cap=0.3, sector_of=sectors, sector_cap=0.35)
    assert w["A"] + w["B"] <= 0.35 + 1e-9 and sum(w.values()) == pytest.approx(1)
    assert cap_weights({"A": -1}, cap=0.5) == {}


def test_first_sessions():
    cal = ["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05", "2026-02-02"]
    months, years = first_sessions(cal)
    assert months == {"2025-12-30", "2026-01-02", "2026-02-02"}
    assert years == {"2025-12-30", "2026-01-02"}


def snap(rows):
    base = dict(open=100.0, close=100.0, vol252=0.2, bars_count=300, mom_score=1.0)
    return pd.DataFrame.from_dict({k: {**base, **v} for k, v in rows.items()}, orient="index")


def test_rule_selection():
    s = snap({"A": dict(vol252=0.1), "B": dict(vol252=0.3), "C": dict(vol252=0.2), "NIFTYBEES": {}})
    core = CoreRules(universe={"A", "B", "C"}, size=2)
    assert core.select(s, {}) == {"A": 0.5, "C": 0.5}
    assert CoreRules(universe=set(), mode="etf").select(s, {}) == {"NIFTYBEES": 1.0}
    sat = SatelliteRules(universe={"A", "B", "C"}, size=2, stock_cap=0.6)
    s2 = snap({"A": dict(mom_score=3.0), "B": dict(mom_score=1.0), "C": dict(mom_score=-1)})
    w = sat.select(s2, {})
    assert set(w) == {"A", "B"} and w["A"] == pytest.approx(0.6)


def book(rules, capital=100_000.0, costs=NO_COSTS):
    return BasketBook(rules, BasketState.new(rules.name, capital), costs, {})


def test_basket_book_select_execute_drift_and_overlay():
    rules = CoreRules(universe={"A", "B"}, size=2)
    b = book(rules)
    months = {"2026-01-01", "2026-02-02"}
    s1 = b.step("2026-01-01", snap({"A": {}, "B": dict(vol252=0.3)}), ON, months, {"2026-01-01"})
    assert s1.reason == "new selection" and {o.symbol: o.qty for o in s1.planned} == {
        "A": 500,
        "B": 500,
    }
    s2 = b.step("2026-01-02", snap({"A": {}, "B": {}}), ON, months, set())
    assert len(s2.executed) == 2 and b.state.cash == pytest.approx(0)
    # A doubles: weights 67/33 -> drift rebalance on the next month start
    s3 = b.step("2026-02-02", snap({"A": dict(open=200, close=200), "B": {}}), ON, months, set())
    assert s3.reason == "weights drifted"
    assert {o.symbol: o.qty for o in s3.planned} == {"A": -125, "B": 250}

    sat = SatelliteRules(universe={"A", "B"}, size=2, stock_cap=0.5)
    b = book(sat)
    b.step("2026-01-01", snap({"A": {}, "B": {}}), ON, months, set())
    b.step("2026-01-02", snap({"A": {}, "B": {}}), ON, months, set())
    s = b.step("2026-02-02", snap({"A": {}, "B": {}}), OFF, months, set())
    assert s.reason.startswith("risk-off") and all(o.qty == -250 for o in s.planned)
    b.step("2026-02-03", snap({"A": {}, "B": {}}), OFF, months, set())
    assert b.state.weights() == {"A": pytest.approx(0.25), "B": pytest.approx(0.25)}


def test_buy_limited_by_cash_with_costs():
    b = book(CoreRules(universe={"A"}, size=1), capital=10_000, costs=CostModel())
    b.step("2026-01-01", snap({"A": {}}), ON, set(), {"2026-01-01"})
    b.step("2026-01-02", snap({"A": dict(open=101.0)}), ON, set(), set())
    assert b.state.cash >= 0 and b.state.holdings["A"] > 0


SYMS = {f"S{i:02d}" for i in range(12)}
UNIVERSE = Universe(SYMS, SYMS, {s: f"Sec{int(s[1:]) % 4}" for s in SYMS}, core=SYMS)


def test_basket_backtest_and_paper(db):
    bars, days = market()
    panel = build_panel(bars, UNIVERSE)
    capital = split_capital(500_000, 70)
    assert capital == {"core": 350_000, "satellite": 150_000}
    rules = build_baskets(UNIVERSE)
    res = run_basket_backtest(
        panel, rules, capital, CostModel(), UNIVERSE.sector_of, days[260], days[-1]
    )
    total = res.sleeves["core"].equity + res.sleeves["satellite"].equity
    assert res.combined.equity.equals(total)
    assert "cagr" in res.combined.metrics and "trades" not in res.combined.metrics

    paper = BasketPaper(db, rules, capital, CostModel(), UNIVERSE.sector_of)
    lines = paper.run(days[300], panel)
    assert any("rebalance at next open" in line for line in lines)
    lines = paper.run(days[301], panel)
    assert any("executed at open" in line for line in lines)
    text = basket_text(paper.all_states(), capital)
    assert "CORE" in text and "SATELLITE" in text
