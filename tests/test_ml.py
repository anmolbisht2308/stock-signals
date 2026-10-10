import numpy as np
import pandas as pd
import pytest

from stock_signals.costs import CostModel
from stock_signals.ml import model as model_mod
from stock_signals.ml.dataset import label_t1_before_stop, positional_dataset, swing_dataset
from stock_signals.ml.evaluate import evaluate, keep_rule, log_decision, train_final
from stock_signals.ml.features import FEATURES, make_features
from stock_signals.ml.model import TrainedModel, train, walk_forward
from stock_signals.ml.scored import LookupScorer, ModelScorer, ScoredStrategy
from stock_signals.models import Candidate
from stock_signals.panel import Universe, build_panel
from stock_signals.regime import Regime
from stock_signals.scan import build_strategies
from stock_signals.sizing import SizingConfig
from tests.test_backtest import market

SYMS = {f"S{i:02d}" for i in range(12)}
U = Universe(SYMS, SYMS, {s: f"Sec{int(s[1:]) % 4}" for s in SYMS})


@pytest.fixture(scope="module")
def mkt():
    bars, days = market()
    return build_panel(bars, U), days


def test_features_same_on_panel_and_snapshot(mkt):
    panel, days = mkt
    full = make_features(panel.data, U.sector_of, panel.regimes)
    d = days[350]
    snap = panel.snapshot(d)
    one = make_features(snap.rename_axis("symbol").reset_index(), U.sector_of, panel.regimes)
    a = full[full["date"] == d].set_index("symbol")[FEATURES].sort_index()
    b = one.set_index("symbol")[FEATURES].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_label_t1_before_stop():
    ohlc = np.array(
        [
            [100, 101, 99, 100],
            [100, 102, 99, 101],  # entry day
            [101, 108, 100, 107],  # T1 hit
        ],
        dtype=float,
    )
    assert label_t1_before_stop(ohlc, 0, 101, 95, 107.5, 15) == 1.0
    assert label_t1_before_stop(ohlc, 0, 101, 99.5, 107.5, 15) == 0.0  # stop first
    assert np.isnan(label_t1_before_stop(ohlc, 0, 99, 95, 107.5, 15))  # gapped above entry
    assert np.isnan(label_t1_before_stop(ohlc, 0, 101, 95, 200, 15))  # future unknown
    assert label_t1_before_stop(ohlc, 0, 101, 95, 200, 2) == 0.0  # time stop


def test_datasets(mkt):
    panel, days = mkt
    swing = build_strategies(U)[0]
    ds = swing_dataset(panel, swing, U.sector_of, days[260], days[-1])
    assert len(ds) > 0 and set(ds["label"].dropna().unique()) <= {0.0, 1.0}
    assert set(FEATURES) <= set(ds.columns)
    pos = positional_dataset(panel, U.sector_of, days[260], days[-1])
    lab = pos["label"].dropna()
    assert len(lab) > 0 and lab.between(0, 1).all()
    assert pos["date"].isin(panel.week_ends).all()


def synthetic_rows(n=600, seed=1):
    rng = np.random.default_rng(seed)
    dates = sorted(pd.bdate_range("2020-01-01", periods=n // 2).strftime("%Y-%m-%d"))
    rows = pd.DataFrame({"date": np.repeat(dates, 2), "symbol": ["A", "B"] * (n // 2)})
    for f in FEATURES:
        rows[f] = rng.normal(size=n)
    rows["label"] = (rows["rsi14"] + rng.normal(scale=0.5, size=n) > 0).astype(float)
    return rows, dates


def test_walk_forward_is_purged(monkeypatch):
    rows, dates = synthetic_rows()
    seen = []
    real_train = model_mod.train

    def spy(df, features, task):
        seen.append(df["date"].max())
        return real_train(df, features, task)

    monkeypatch.setattr(model_mod, "train", spy)
    first_test = dates[150]
    preds = walk_forward(
        rows, FEATURES, "binary", dates, first_test, embargo=16, block=50, min_train_rows=100
    )
    assert preds[rows["date"] < first_test].isna().all()
    assert preds[rows["date"] >= first_test].notna().any()
    pos = {d: i for i, d in enumerate(dates)}
    for b, max_train in zip(range(150, 300, 50), seen, strict=False):
        assert pos[max_train] < b - 16 + 1
    # the signal is learnable: OOS accuracy above chance
    oos = rows[preds.notna()]
    acc = ((preds[preds.notna()] > 0.5) == (oos["label"] == 1)).mean()
    assert acc > 0.6


def test_model_save_load(tmp_path):
    rows, _ = synthetic_rows()
    m = train(rows, FEATURES, "binary")
    m.threshold = 0.42
    path = tmp_path / "models" / "swing-lgbm.txt"
    m.save(path)
    again = TrainedModel.load(path)
    assert again.threshold == 0.42 and again.features == FEATURES
    np.testing.assert_allclose(again.predict(rows), m.predict(rows))


def cand(sym, score=0.0):
    return Candidate(
        symbol=sym, ref_price=100, entry_low=99, entry_high=101, stop=95, score=score, reason="rule"
    )


class Fixed:
    book, version = "swing", "swing-v1"

    def candidates(self, snap, regime, week_end):
        return [cand("A"), cand("B"), cand("C")]

    def max_new(self, regime):
        return 5

    def close_exit(self, pos, row, week_end):
        return None


def test_scored_strategy_filters_and_ranks():
    scores = pd.Series(
        {("2026-01-02", "A"): 0.4, ("2026-01-02", "B"): 0.9, ("2026-01-02", "C"): 0.6}
    )
    s = ScoredStrategy(Fixed(), LookupScorer(scores), threshold=0.5)
    snap = pd.DataFrame({"date": ["2026-01-02"] * 3}, index=["A", "B", "C"])
    out = s.candidates(snap, Regime("d", "risk_on", 1, 0, 0.5), False)
    assert [c.symbol for c in out] == ["B", "C"] and s.version == "swing-v1+lgbm"
    assert "model 0.90" in out[0].reason


def test_keep_rule():
    base = {"expectancy_r": 0.10, "max_drawdown": -0.10, "trades": 100}
    assert keep_rule(base, {"expectancy_r": 0.2, "max_drawdown": -0.11, "trades": 50})[0]
    ok, why = keep_rule(base, {"expectancy_r": 0.2, "max_drawdown": -0.13, "trades": 50})
    assert not ok and "drawdown" in why[0]
    assert not keep_rule(base, {"expectancy_r": 0.05, "max_drawdown": -0.05, "trades": 50})[0]
    assert not keep_rule(base, {"expectancy_r": 0.5, "max_drawdown": -0.05, "trades": 5})[0]


def test_evaluate_and_train_final_end_to_end(mkt, tmp_path):
    panel, days = mkt
    swing = build_strategies(U)[0]
    ev = evaluate(
        panel,
        swing,
        200_000,
        CostModel(),
        SizingConfig(),
        U.sector_of,
        days[0],
        {"test": (days[330], days[-1])},
        "test",
    )
    assert ev.variant == "swing-v1+lgbm" and "test" in ev.windows
    assert ev.reasons  # KEEP or DROP with reasons
    log_decision(tmp_path / "model-decisions.md", ev)
    assert "swing-v1+lgbm" in (tmp_path / "model-decisions.md").read_text()
    m = train_final(panel, swing, U.sector_of, days[0], days[-1])
    assert 0 < m.threshold < 1
    scorer = ModelScorer(m, U.sector_of, panel.regimes)
    snap = panel.snapshot(days[-1])
    got = scorer.score(days[-1], snap, sorted(SYMS)[:3])
    assert set(got) == set(sorted(SYMS)[:3])
