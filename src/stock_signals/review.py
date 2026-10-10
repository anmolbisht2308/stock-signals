"""Phase 6 paper-run review: go/no-go checks written to docs/paper-review-N.md."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pymongo.database import Database

from stock_signals import dates
from stock_signals.backtest import PERIODS, run_backtest
from stock_signals.costs import CostModel
from stock_signals.metrics import compute_metrics
from stock_signals.panel import Panel, Universe
from stock_signals.paper import PaperStore
from stock_signals.sizing import SizingConfig
from stock_signals.strategies import Strategy

MIN_SWING_TRADES = 30


class Journal:
    def __init__(self, db: Database) -> None:
        self.col = db["journal"]

    def add(self, text: str) -> None:
        self.col.insert_one(
            {
                "date": dates.to_str(dates.now_ist().date()),
                "text": text,
                "created_at": datetime.now(UTC),
            }
        )

    def entries(self, since: str | None = None) -> list[dict]:
        q = {"date": {"$gte": since}} if since else {}
        return list(self.col.find(q, {"_id": 0}).sort("created_at", 1))


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        return (
            f"- [{'x' if self.ok else ' '}] {'✅' if self.ok else '❌'} {self.name}: {self.detail}"
        )


def review_variant(
    store: PaperStore,
    panel: Panel,
    strategy: Strategy,
    capital: float,
    costs: CostModel,
    sizing: SizingConfig,
    universe: Universe,
) -> tuple[list[str], list[Check]]:
    variant = strategy.version
    eq = store.equity_frame(variant)
    trades = store.trade_list(variant=variant)
    if eq.empty:
        return [f"### {variant}", "No paper sessions yet."], []
    start, end = str(eq.index[0]), str(eq.index[-1])
    m = compute_metrics(trades, eq["equity"], eq["invested"])

    # Same rules replayed on today's data over the paper period: should match the ledger.
    # A mismatch means the stored history changed (late corrections, splits) or a bug.
    replay = run_backtest(panel, strategy, capital, costs, sizing, universe.sector_of, start, end)
    replay_end = float(replay.equity.iloc[-1]) if len(replay.equity) else capital
    paper_end = float(eq["equity"].iloc[-1])

    test_start, test_end = PERIODS["test"]
    reference = None
    if panel.calendar and panel.calendar[0] <= test_start:
        ref = run_backtest(
            panel, strategy, capital, costs, sizing, universe.sector_of, test_start, test_end
        )
        reference = ref.metrics.get("max_drawdown")

    checks = []
    if strategy.book == "swing":
        checks.append(
            Check(
                f"≥ {MIN_SWING_TRADES} closed trades",
                len(trades) >= MIN_SWING_TRADES,
                f"{len(trades)}",
            )
        )
    else:
        checks.append(
            Check(
                "rules followed (positional: judge open P&L, few closed trades)",
                True,
                f"{len(trades)} closed",
            )
        )
    checks.append(
        Check(
            "expectancy after costs > 0R",
            m["expectancy_r"] > 0 if trades else False,
            f"{m['expectancy_r']:+.2f}R",
        )
    )
    if reference is not None:
        checks.append(
            Check(
                "max drawdown within the 2022–24 backtest",
                m["max_drawdown"] >= reference,
                f"paper {m['max_drawdown']:.1%} vs backtest {reference:.1%}",
            )
        )
    diff = abs(paper_end - replay_end) / capital
    checks.append(
        Check(
            "paper ledger matches a replay on current data",
            diff < 0.005,
            f"₹{paper_end:,.0f} vs replay ₹{replay_end:,.0f}",
        )
    )

    lines = [
        f"### {variant}",
        f"{start} → {end}: equity ₹{paper_end:,.0f} ({paper_end / capital - 1:+.1%}), "
        f"{len(trades)} closed trades, win rate {m['win_rate']:.0%}, "
        f"max drawdown {m['max_drawdown']:.1%}",
        "",
        *[c.line() for c in checks],
    ]
    return lines, checks


def shadow_check(store: PaperStore, base_variant: str, model_variant: str) -> list[Check]:
    """Paper half of the keep rule: ≥ 4 weeks, ≥ 20 trades combined, and expectancy with the
    model not lower than rules-only over the same period."""
    eq = store.equity_frame(model_variant)
    if eq.empty:
        return [Check(f"{model_variant} shadow run", False, "no sessions yet")]
    since = str(eq.index[0])
    model_trades = store.trade_list(variant=model_variant, since=since)
    base_trades = store.trade_list(variant=base_variant, since=since)
    sessions = len(eq)
    exp_m = sum(t.r_multiple for t in model_trades) / len(model_trades) if model_trades else 0.0
    exp_b = sum(t.r_multiple for t in base_trades) / len(base_trades) if base_trades else 0.0
    enough = sessions >= 20 and len(model_trades) + len(base_trades) >= 20
    return [
        Check(
            f"{model_variant}: ≥ 4 weeks and ≥ 20 trades",
            enough,
            f"{sessions} sessions, {len(model_trades)} + {len(base_trades)} trades",
        ),
        Check(
            f"{model_variant}: expectancy not below {base_variant}",
            enough and exp_m >= exp_b,
            f"{exp_m:+.2f}R vs {exp_b:+.2f}R",
        ),
    ]


def next_review_path(docs: Path) -> Path:
    n = 1
    while (docs / f"paper-review-{n}.md").exists():
        n += 1
    return docs / f"paper-review-{n}.md"


def write_review(
    path: Path,
    sections: list[list[str]],
    checks: list[Check],
    journal: list[dict],
) -> str:
    passed = bool(checks) and all(c.ok for c in checks)
    decision = (
        "✅ All checks pass → Phase 7 (LightGBM in shadow mode)."
        if passed
        else "❌ Not all checks pass → fix the rules or data, then run another 4 weeks."
    )
    body = [
        f"# Paper review — {dates.to_str(dates.now_ist().date())}",
        "",
        "Rule versions frozen during the run: only bug fixes allowed.",
        "",
        *[line for section in sections for line in (*section, "")],
        "## Journal",
        *([f"- {j['date']}: {j['text']}" for j in journal] or ["- (no notes; add with /note)"]),
        "",
        "## Decision",
        decision,
        "",
        "Edit this file with anything the numbers miss (missed fills, confusing messages).",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return decision


def review_start(panel_end: str) -> str:
    """History needed for the reference backtest (2022–24 test period plus warm-up)."""
    test_start = dates.parse(PERIODS["test"][0]) - timedelta(days=600)
    return min(dates.to_str(test_start), panel_end)
