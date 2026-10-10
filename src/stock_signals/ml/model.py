"""LightGBM training, purged walk-forward predictions, saving and loading."""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
    "deterministic": True,
    "num_threads": 2,
}
ROUNDS = 300


@dataclass
class TrainedModel:
    booster: lgb.Booster
    features: list[str]
    task: str  # "binary" (swing) or "regression" (positional)
    train_end: str
    threshold: float = 0.5

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.features].astype(float))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        meta = {
            "features": self.features,
            "task": self.task,
            "train_end": self.train_end,
            "threshold": self.threshold,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(
            lgb.Booster(model_file=str(path)),
            meta["features"],
            meta["task"],
            meta["train_end"],
            meta.get("threshold", 0.5),
        )


def train(df: pd.DataFrame, features: list[str], task: str) -> TrainedModel:
    rows = df.dropna(subset=["label"])
    params = {**PARAMS, "objective": task}
    ds = lgb.Dataset(rows[features].astype(float), label=rows["label"].astype(float))
    booster = lgb.train(params, ds, num_boost_round=ROUNDS)
    return TrainedModel(booster, features, task, str(rows["date"].max()))


def walk_forward(
    df: pd.DataFrame,
    features: list[str],
    task: str,
    calendar: list[str],
    first_test: str,
    embargo: int,
    block: int = 126,
    min_train_rows: int = 200,
) -> pd.Series:
    """Out-of-sample predictions indexed like `df`. Each block of `block` sessions is
    predicted by a model trained only on rows whose label window ended before the block:
    rows dated more than `embargo` sessions before the block start (purging)."""
    preds = pd.Series(np.nan, index=df.index)
    sessions = [d for d in calendar if d >= first_test]
    pos = {d: i for i, d in enumerate(calendar)}
    for b in range(0, len(sessions), block):
        block_start = sessions[b]
        block_end = sessions[min(b + block, len(sessions)) - 1]
        cutoff_i = pos[block_start] - embargo
        if cutoff_i <= 0:
            continue
        cutoff = calendar[cutoff_i]
        train_rows = df[(df["date"] < cutoff) & df["label"].notna()]
        test_mask = (df["date"] >= block_start) & (df["date"] <= block_end)
        if len(train_rows) < min_train_rows or not test_mask.any():
            continue
        model = train(train_rows, features, task)
        preds[test_mask] = model.predict(df[test_mask])
    return preds
