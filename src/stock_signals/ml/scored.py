"""Strategy wrapper that filters and re-ranks rule candidates with a model score.

The model never creates signals: it only keeps the rule candidates it scores above a
threshold and orders them by score. Exits are the rules' exits, unchanged.
"""

from typing import Protocol

import pandas as pd

from stock_signals.ml.features import make_features
from stock_signals.ml.model import TrainedModel
from stock_signals.models import Candidate, Position
from stock_signals.regime import Regime
from stock_signals.strategies import Strategy


class Scorer(Protocol):
    def score(self, date: str, snap: pd.DataFrame, symbols: list[str]) -> dict[str, float]: ...


class LookupScorer:
    """Precomputed (date, symbol) → score, e.g. walk-forward out-of-sample predictions."""

    def __init__(self, scores: pd.Series) -> None:
        self.scores = scores  # MultiIndex (date, symbol)

    def score(self, date: str, snap: pd.DataFrame, symbols: list[str]) -> dict[str, float]:
        out = {}
        for s in symbols:
            v = self.scores.get((date, s))
            if v is not None and v == v:
                out[s] = float(v)
        return out


class ModelScorer:
    """Scores live snapshots with a saved model."""

    def __init__(
        self,
        model: TrainedModel,
        sector_of: dict[str, str],
        regimes: pd.DataFrame,
        extra: pd.DataFrame | None = None,
    ) -> None:
        self.model = model
        self.sector_of = sector_of
        self.regimes = regimes
        self.extra = extra

    def score(self, date: str, snap: pd.DataFrame, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        df = snap.rename_axis("symbol").reset_index()
        df["date"] = date
        feats = make_features(df, self.sector_of, self.regimes, self.extra)
        feats = feats[feats["symbol"].isin(symbols)]
        if feats.empty:
            return {}
        return dict(zip(feats["symbol"], self.model.predict(feats), strict=True))


class ScoredStrategy:
    def __init__(
        self, inner: Strategy, scorer: Scorer, threshold: float, suffix: str = "lgbm"
    ) -> None:
        self.inner = inner
        self.scorer = scorer
        self.threshold = threshold
        self.book = inner.book
        self.version = f"{inner.version}+{suffix}"

    def candidates(self, snap: pd.DataFrame, regime: Regime, week_end: bool) -> list[Candidate]:
        cands = self.inner.candidates(snap, regime, week_end)
        if not cands:
            return []
        date = str(snap["date"].iloc[0])
        scores = self.scorer.score(date, snap, [c.symbol for c in cands])
        kept = []
        for c in cands:
            s = scores.get(c.symbol)
            if s is None or s < self.threshold:
                continue
            kept.append(
                c.model_copy(
                    update={
                        "score": s,
                        "reason": f"{c.reason}, model {s:.2f}",
                    }
                )
            )
        return sorted(kept, key=lambda c: -c.score)

    def max_new(self, regime: Regime) -> int:
        return self.inner.max_new(regime)

    def close_exit(self, pos: Position, row: pd.Series, week_end: bool) -> str | None:
        return self.inner.close_exit(pos, row, week_end)
