import numpy as np
import pandas as pd
import pytest

from stock_signals.bars import BarStore
from stock_signals.ml.kronos import (
    COLUMNS,
    KronosCache,
    KronosFeatures,
    KronosSampler,
    kronos_report,
    path_features,
    seed_for,
)
from stock_signals.panel import Universe, build_panel
from tests.test_backtest import market


def test_path_features():
    # 4 paths, 3 days, OHLC. close 100, ATR 2 -> stop 97, T1 104.5
    up = [[100, 101, 99, 100], [100, 105, 99, 104], [104, 106, 103, 105]]  # T1 day 2
    down = [[100, 100, 96, 97], [97, 110, 96, 109], [109, 110, 108, 109]]  # stop first
    flat = [[100, 101, 99, 100], [100, 101, 99, 100], [100, 101, 99, 99]]
    paths = np.array([up, down, flat, up], dtype=float)
    f = path_features(paths, close=100.0, atr=2.0)
    assert f["kronos_p_up"] == 0.75
    assert f["kronos_p_t1"] == 0.5
    assert f["kronos_exp_ret"] == pytest.approx(np.mean([0.05, 0.09, -0.01, 0.05]))


class FakeSampler:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def sample(self, history, horizon, n, seed):
        self.calls += 1
        rng = np.random.default_rng(seed)
        last = float(history["close"].iloc[-1])
        steps = np.cumprod(1 + rng.normal(0.001, 0.02, size=(n, horizon)), axis=1) * last
        return np.stack([steps, steps * 1.01, steps * 0.99, steps], axis=-1)


def test_provider_computes_once_and_caches(db):
    bars, days = market()
    db["bars"].insert_many(bars.to_dict("records"))
    syms = {f"S{i:02d}" for i in range(12)}
    panel = build_panel(bars, Universe(syms, syms, {}))
    d = days[350]
    snap = panel.snapshot(d)
    sampler = FakeSampler()
    feats = KronosFeatures(KronosCache(db), BarStore(db), sampler, horizon=10, samples=20)
    out = feats.get(d, ["S01", "S02"], snap)
    assert set(out.columns) == set(COLUMNS) and len(out) == 2 and sampler.calls == 2
    again = feats.get(d, ["S01", "S02"], snap)
    assert sampler.calls == 2
    pd.testing.assert_frame_equal(out, again)
    no_sampler = KronosFeatures(KronosCache(db), BarStore(db), None)
    assert len(no_sampler.get(d, ["S03"], snap)) == 2  # cached rows only, nothing computed
    assert seed_for("S01", d) == seed_for("S01", d) != seed_for("S02", d)

    report = kronos_report(KronosCache(db).frame(), panel.data, horizon=10, start=days[0])
    assert report[0].startswith("🔮") and "always-up" in report[1]


def test_sampler_requires_repo(tmp_path):
    with pytest.raises(RuntimeError, match="Kronos repo not found"):
        KronosSampler(tmp_path)


FAKE_MODEL = """
import numpy as np
import pandas as pd


class _Loadable:
    @classmethod
    def from_pretrained(cls, name):
        return cls()


class Kronos(_Loadable):
    pass


class KronosTokenizer(_Loadable):
    pass


class KronosPredictor:
    # Same signatures as the real repo (model/kronos.py).
    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        self.max_context = max_context

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9,
                sample_count=1, verbose=True):
        raise AssertionError("predict_batch should be used")

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0,
                      top_k=0, top_p=0.9, sample_count=1, verbose=True):
        assert sample_count == 1 and verbose is False
        out = []
        for i, (df, xs, ys) in enumerate(zip(df_list, x_timestamp_list, y_timestamp_list)):
            assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]
            assert len(df) == len(xs) <= self.max_context and len(ys) == pred_len
            last = df["close"].iloc[-1] * (1 + 0.01 * i)
            vals = np.tile([last, last * 1.01, last * 0.99, last, 1.0, 1.0], (pred_len, 1))
            out.append(pd.DataFrame(vals, columns=["open", "high", "low", "close", "volume",
                                                   "amount"], index=ys))
        return out
"""


def test_sampler_calls_kronos_batch_api(tmp_path, monkeypatch):
    import sys
    import types

    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "__init__.py").write_text(FAKE_MODEL)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(manual_seed=lambda s: None))
    monkeypatch.delitem(sys.modules, "model", raising=False)
    sampler = KronosSampler(tmp_path, max_context=50)
    days = pd.bdate_range("2025-01-01", periods=80).strftime("%Y-%m-%d")
    hist = pd.DataFrame(
        {"date": days, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e5}
    )
    paths = sampler.sample(hist, horizon=5, n=3, seed=1)
    assert paths.shape == (3, 5, 4)
    assert paths[2, 0, 3] == pytest.approx(102.0)
    sys.path.remove(str(tmp_path))
