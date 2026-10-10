"""Rules vs rules + model on out-of-sample predictions, and the keep rule."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from stock_signals.backtest import BacktestResult, run_backtest
from stock_signals.costs import CostModel
from stock_signals.ml.dataset import POSITIONAL_HORIZON, positional_dataset, swing_dataset
from stock_signals.ml.features import FEATURES, KRONOS
from stock_signals.ml.model import TrainedModel, train, walk_forward
from stock_signals.ml.scored import LookupScorer, ScoredStrategy
from stock_signals.panel import Panel
from stock_signals.sizing import SizingConfig
from stock_signals.strategies import Strategy

MIN_MODEL_TRADES = 20
MAX_DD_WORSEN = 0.20


def keep_rule(
    base: dict, model: dict, min_trades: int = MIN_MODEL_TRADES
) -> tuple[bool, list[str]]:
    """Backtest half of the keep rule (docs/PLAN.md): higher expectancy after costs and a max
    drawdown no more than 20% worse than without the model."""
    reasons = []
    ok = True
    if model.get("trades", 0) < min_trades:
        ok = False
        reasons.append(f"only {model.get('trades', 0):.0f} trades with the model (< {min_trades})")
    if model.get("expectancy_r", 0) <= base.get("expectancy_r", 0):
        ok = False
        reasons.append(
            f"expectancy {model.get('expectancy_r', 0):+.2f}R not above "
            f"{base.get('expectancy_r', 0):+.2f}R"
        )
    base_dd, model_dd = base.get("max_drawdown", 0), model.get("max_drawdown", 0)
    if model_dd < base_dd * (1 + MAX_DD_WORSEN):
        ok = False
        reasons.append(f"drawdown {model_dd:.1%} worse than {base_dd:.1%} by > 20%")
    if ok:
        reasons.append(
            f"expectancy {base.get('expectancy_r', 0):+.2f}R → "
            f"{model.get('expectancy_r', 0):+.2f}R, drawdown {base_dd:.1%} → {model_dd:.1%}"
        )
    return ok, reasons


@dataclass
class Evaluation:
    book: str
    variant: str
    windows: dict[str, tuple[BacktestResult, BacktestResult]] = field(default_factory=dict)
    keep: bool = False
    reasons: list[str] = field(default_factory=list)
    threshold: float = 0.5
    oos_rows: int = 0

    def lines(self) -> list[str]:
        out = [
            f"🧪 {self.variant} ({self.oos_rows} out-of-sample rows, "
            f"threshold {self.threshold:.2f})"
        ]
        for name, (base, model) in self.windows.items():
            b, m = base.metrics, model.metrics
            out.append(
                f"{name} {base.start} → {base.end}: rules {b.get('expectancy_r', 0):+.2f}R × "
                f"{b.get('trades', 0):.0f} trades, DD {b.get('max_drawdown', 0):.1%} | "
                f"+model {m.get('expectancy_r', 0):+.2f}R × {m.get('trades', 0):.0f}, "
                f"DD {m.get('max_drawdown', 0):.1%}"
            )
        out.append(("✅ KEEP: " if self.keep else "❌ DROP: ") + "; ".join(self.reasons))
        return out


def feature_list(use_kronos: bool) -> list[str]:
    return FEATURES + (KRONOS if use_kronos else [])


def build_dataset(
    panel: Panel,
    strategy: Strategy,
    sector_of: dict[str, str],
    start: str,
    end: str,
    extra: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str, int]:
    """(rows, task, embargo sessions)"""
    if strategy.book == "swing":
        ds = swing_dataset(panel, strategy, sector_of, start, end, extra)
        horizon = getattr(strategy, "time_stop_days", 15)
        return ds, "binary", horizon + 1
    ds = positional_dataset(panel, sector_of, start, end, extra)
    return ds, "regression", POSITIONAL_HORIZON + 1


def evaluate(
    panel: Panel,
    strategy: Strategy,
    capital: float,
    costs: CostModel,
    sizing: SizingConfig,
    sector_of: dict[str, str],
    data_start: str,
    windows: dict[str, tuple[str, str]],
    decide_on: str,
    extra: pd.DataFrame | None = None,
    suffix: str = "lgbm",
) -> Evaluation:
    first_test = min(w[0] for w in windows.values())
    end = max(w[1] for w in windows.values())
    features = feature_list(extra is not None)
    ds, task, embargo = build_dataset(panel, strategy, sector_of, data_start, end, extra)
    variant = f"{strategy.version}+{suffix}"
    ev = Evaluation(book=strategy.book, variant=variant)
    if ds.empty:
        ev.reasons = ["no training rows"]
        return ev
    preds = walk_forward(ds, features, task, panel.calendar, first_test, embargo)
    train_rows = ds[(ds["date"] < first_test) & ds["label"].notna()]
    ev.threshold = (
        float(train_rows["label"].mean()) if task == "binary" and len(train_rows) else 0.5
    )
    scores = pd.Series(preds.values, index=pd.MultiIndex.from_arrays([ds["date"], ds["symbol"]]))
    scores = scores[~scores.index.duplicated()]
    ev.oos_rows = int(preds.notna().sum())
    scored = ScoredStrategy(strategy, LookupScorer(scores), ev.threshold, suffix)
    snapshots = panel.by_date()
    for name, (start, stop) in windows.items():
        base = run_backtest(
            panel, strategy, capital, costs, sizing, sector_of, start, stop, snapshots
        )
        model = run_backtest(
            panel, scored, capital, costs, sizing, sector_of, start, stop, snapshots
        )
        ev.windows[name] = (base, model)
    base, model = ev.windows[decide_on]
    ev.keep, ev.reasons = keep_rule(base.metrics, model.metrics)
    return ev


def train_final(
    panel: Panel,
    strategy: Strategy,
    sector_of: dict[str, str],
    data_start: str,
    end: str,
    threshold: float | None = None,
    extra: pd.DataFrame | None = None,
) -> TrainedModel:
    """Train on all labelled history. Default threshold: the base win rate (swing) or the
    median rank 0.5 (positional) — keep signals the model rates better than average."""
    ds, task, _ = build_dataset(panel, strategy, sector_of, data_start, end, extra)
    model = train(ds, feature_list(extra is not None), task)
    if threshold is None:
        labelled = ds["label"].dropna()
        threshold = float(labelled.mean()) if task == "binary" and len(labelled) else 0.5
    model.threshold = threshold
    return model


def log_decision(path: Path, ev: Evaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Model decisions\n\nAppended by `stock-signals ml-evaluate`.\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {date.today().isoformat()} — {ev.variant}\n\n")
        f.write("\n".join(f"- {line}" for line in ev.lines()) + "\n")
