"""Kronos (github.com/shiyu-coder/Kronos, MIT) as an extra feature source.

Kronos is not on PyPI. Clone the repo and point KRONOS_PATH at it:

    git clone https://github.com/shiyu-coder/Kronos ~/Kronos
    uv pip install torch einops huggingface_hub safetensors   # CPU torch is fine

Weights download from Hugging Face on first use (NeoQuasar/Kronos-small + tokenizer).

We sample many future paths per stock (Kronos' `predict` averages samples, so we call it
with `sample_count=1` repeatedly) and turn them into four features:

    kronos_p_up      share of paths closing above today's close at the horizon
    kronos_exp_ret   mean return at the horizon
    kronos_spread    std of returns at the horizon
    kronos_p_t1      share of paths touching the swing T1 before the stop
                     (stop = close − 1.5×ATR, T1 = close + 1.5R, same as swing-v1)

Zero-shot inference only normalises with the context window, so the look-ahead bug in the
repo's fine-tuning scripts does not apply here. Kronos was pre-trained on data up to about
mid-2024, so it is only evaluated on sessions from EVAL_START on.
"""

import hashlib
import inspect
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from pymongo import ASCENDING
from pymongo.database import Database

from stock_signals.bars import BarStore

log = logging.getLogger(__name__)

EVAL_START = "2024-07-01"
COLUMNS = ["kronos_p_up", "kronos_exp_ret", "kronos_spread", "kronos_p_t1"]
ATR_STOP, T1_R = 1.5, 1.5


class PathSampler(Protocol):
    name: str

    def sample(self, history: pd.DataFrame, horizon: int, n: int, seed: int) -> np.ndarray:
        """`history`: date, open, high, low, close, volume (oldest first).
        Returns paths shaped (n, horizon, 4) with OHLC."""
        ...


class KronosSampler:
    def __init__(
        self,
        repo_path: Path,
        model: str = "NeoQuasar/Kronos-small",
        tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cpu",
        max_context: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> None:
        repo_path = Path(repo_path).expanduser()
        if not (repo_path / "model").exists():
            raise RuntimeError(
                f"Kronos repo not found at {repo_path}: git clone "
                "https://github.com/shiyu-coder/Kronos and set KRONOS_PATH"
            )
        sys.path.insert(0, str(repo_path))
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore[import-not-found]

        self.torch = torch
        self.name = model.split("/")[-1]
        self.predictor = KronosPredictor(
            Kronos.from_pretrained(model),
            KronosTokenizer.from_pretrained(tokenizer),
            device=device,
            max_context=max_context,
        )
        self.max_context = max_context
        self.temperature = temperature
        self.top_p = top_p
        params = inspect.signature(self.predictor.predict).parameters
        self._extra = {"verbose": False} if "verbose" in params else {}

    def sample(self, history: pd.DataFrame, horizon: int, n: int, seed: int) -> np.ndarray:
        h = history.tail(self.max_context).reset_index(drop=True)
        x = h[["open", "high", "low", "close", "volume"]].astype(float).copy()
        x["amount"] = x["close"] * x["volume"]
        x_ts = pd.Series(pd.to_datetime(h["date"]))
        y_ts = pd.Series(pd.bdate_range(x_ts.iloc[-1] + pd.Timedelta(days=1), periods=horizon))
        self.torch.manual_seed(seed)
        common = dict(
            pred_len=horizon, T=self.temperature, top_p=self.top_p, sample_count=1, **self._extra
        )
        if hasattr(self.predictor, "predict_batch"):
            # The same series n times with sample_count=1 = n independent paths in one pass.
            preds = self.predictor.predict_batch(
                df_list=[x] * n,
                x_timestamp_list=[x_ts] * n,
                y_timestamp_list=[y_ts] * n,
                **common,
            )
        else:
            preds = [
                self.predictor.predict(df=x, x_timestamp=x_ts, y_timestamp=y_ts, **common)
                for _ in range(n)
            ]
        return np.stack([p[["open", "high", "low", "close"]].to_numpy(dtype=float) for p in preds])


def path_features(paths: np.ndarray, close: float, atr: float) -> dict[str, float]:
    rets = paths[:, -1, 3] / close - 1
    stop = close - ATR_STOP * atr
    t1 = close + T1_R * (close - stop)
    hits = 0
    for p in paths:
        for _, high, low, _ in p:
            if low <= stop:
                break
            if high >= t1:
                hits += 1
                break
    return {
        "kronos_p_up": float((rets > 0).mean()),
        "kronos_exp_ret": float(rets.mean()),
        "kronos_spread": float(rets.std()),
        "kronos_p_t1": hits / len(paths),
    }


def seed_for(symbol: str, date: str) -> int:
    return int(hashlib.sha1(f"{symbol}:{date}".encode()).hexdigest()[:8], 16)


class KronosCache:
    def __init__(self, db: Database) -> None:
        self.col = db["kronos_features"]

    def ensure_indexes(self) -> None:
        self.col.create_index([("date", ASCENDING), ("symbol", ASCENDING)], unique=True)

    def put(
        self,
        date: str,
        symbol: str,
        feats: dict[str, float],
        model: str,
        horizon: int,
        samples: int,
    ) -> None:
        self.col.update_one(
            {"date": date, "symbol": symbol},
            {
                "$set": {
                    **feats,
                    "model": model,
                    "horizon": horizon,
                    "samples": samples,
                    "created_at": datetime.now(UTC),
                }
            },
            upsert=True,
        )

    def frame(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        q: dict = {}
        if start or end:
            q["date"] = {k: v for k, v in (("$gte", start), ("$lte", end)) if v}
        docs = list(self.col.find(q, {"_id": 0, "date": 1, "symbol": 1, **{c: 1 for c in COLUMNS}}))
        if not docs:
            return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(docs).set_index(["date", "symbol"])[COLUMNS].sort_index()


class KronosFeatures:
    """Cached Kronos features for (date, symbols); computes missing ones if a sampler is set."""

    def __init__(
        self,
        cache: KronosCache,
        bars: BarStore,
        sampler: PathSampler | None,
        horizon: int = 15,
        samples: int = 30,
        lookback: int = 400,
    ) -> None:
        self.cache = cache
        self.bars = bars
        self.sampler = sampler
        self.horizon = horizon
        self.samples = samples
        self.lookback = lookback

    def get(self, date: str, symbols: list[str], snap: pd.DataFrame) -> pd.DataFrame:
        have = self.cache.frame(start=date, end=date)
        missing = [s for s in symbols if (date, s) not in have.index]
        if missing and self.sampler is not None:
            self.cache.ensure_indexes()
            for sym in missing:
                if sym not in snap.index or not snap.at[sym, "atr14"] > 0:
                    continue
                history = self.bars.load(sym, end=date, last_n=self.lookback)
                if len(history) < 100:
                    continue
                paths = self.sampler.sample(
                    history, self.horizon, self.samples, seed_for(sym, date)
                )
                feats = path_features(
                    paths, float(snap.at[sym, "close"]), float(snap.at[sym, "atr14"])
                )
                self.cache.put(date, sym, feats, self.sampler.name, self.horizon, self.samples)
            have = self.cache.frame(start=date, end=date)
        return have


def kronos_report(
    cache: pd.DataFrame, panel_data: pd.DataFrame, horizon: int = 15, start: str = EVAL_START
) -> list[str]:
    """Honest check against simple baselines on cached rows from `start` on."""
    d = panel_data.sort_values(["symbol", "date"])
    fwd = d.groupby("symbol", sort=False)["close"].shift(-horizon) / d["close"] - 1
    truth = pd.DataFrame(
        {"fwd": fwd.values, "roc126": d["roc126"].values},
        index=pd.MultiIndex.from_arrays([d["date"], d["symbol"]]),
    )
    rows = (
        cache[cache.index.get_level_values(0) >= start]
        .join(truth, how="inner")
        .dropna(subset=["fwd"])
    )
    if rows.empty:
        return [f"No Kronos rows with known {horizon}-day outcomes since {start}"]
    up = rows["fwd"] > 0
    acc_k = float(((rows["kronos_p_up"] > 0.5) == up).mean())
    acc_up = float(up.mean())
    acc_mom = float(((rows["roc126"] > 0) == up).mean())
    ic_k = rows["kronos_exp_ret"].corr(rows["fwd"], method="spearman")
    ic_m = rows["roc126"].corr(rows["fwd"], method="spearman")
    return [
        f"🔮 Kronos vs baselines: {len(rows)} rows since {start}, {horizon}-day horizon",
        f"Direction accuracy: Kronos {acc_k:.1%} | always-up {acc_up:.1%} | "
        f"6m momentum {acc_mom:.1%}",
        f"Rank IC (Spearman): Kronos expected return {ic_k:+.3f} | 6m momentum {ic_m:+.3f}",
        "Kronos adds value only if it beats both baselines; otherwise drop it (keep rule).",
    ]
