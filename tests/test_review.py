from stock_signals.bars import BarStore
from stock_signals.costs import CostModel
from stock_signals.panel import Universe, build_panel
from stock_signals.paper import PaperStore, PaperTrader
from stock_signals.review import Journal, next_review_path, review_variant, write_review
from stock_signals.scan import build_strategies
from stock_signals.sizing import SizingConfig
from tests.test_backtest import market

SYMS = {f"S{i:02d}" for i in range(12)}
U = Universe(SYMS, SYMS, {s: f"Sec{int(s[1:]) % 4}" for s in SYMS})


def test_review_checks_and_file(db, tmp_path):
    bars, days = market()
    panel = build_panel(bars, U)
    strategies = build_strategies(U)
    t = PaperTrader(
        PaperStore(db),
        BarStore(db),
        U,
        strategies,
        {"swing": 200_000, "positional": 300_000},
        CostModel(),
        SizingConfig(),
    )
    for d in days[260:400]:
        t.run(d, panel)
    lines, checks = review_variant(
        t.store, panel, strategies[0], 200_000, CostModel(), SizingConfig(), U
    )
    names = [c.name for c in checks]
    assert "paper ledger matches a replay on current data" in names
    assert next(c for c in checks if "replay" in c.name).ok
    Journal(db).add("missed fill on S03")
    path = next_review_path(tmp_path)
    assert path.name == "paper-review-1.md"
    decision = write_review(path, [lines], checks, Journal(db).entries())
    text = path.read_text()
    assert "swing-v1" in text and "missed fill on S03" in text and decision in text
    path2 = next_review_path(tmp_path)
    assert path2.name == "paper-review-2.md"
