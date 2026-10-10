import numpy as np
import pandas as pd

from stock_signals.backtest import run_backtest
from stock_signals.costs import CostModel
from stock_signals.metrics import compute_metrics, max_drawdown, metrics_lines
from stock_signals.panel import Universe, build_panel
from stock_signals.report import render_html, write_report
from stock_signals.scan import build_strategies
from stock_signals.sizing import SizingConfig
from tests.factories import calendar, series, trend

N = 420


def market():
    days = calendar(N)
    frames = [series("NIFTY50", trend(N, daily=0.0006, wiggle=0.01), days, volume=0)]
    rng = np.random.default_rng(7)
    for i in range(12):
        closes = trend(N, daily=0.0004 + 0.0002 * i, wiggle=0.03, phase=i)
        closes = closes * np.exp(np.cumsum(rng.normal(0, 0.012, N)))
        vol = rng.uniform(1.5e6, 4e6, N)
        frames.append(series(f"S{i:02d}", closes, days, volume=vol))
    return pd.concat(frames), days


def test_backtest_runs_and_is_consistent(tmp_path):
    bars, days = market()
    syms = {f"S{i:02d}" for i in range(12)}
    universe = Universe(
        swing=syms, positional=syms, sector_of={s: f"Sec{int(s[1:]) % 4}" for s in syms}
    )
    panel = build_panel(bars, universe)
    strategies = {s.book: s for s in build_strategies(universe)}
    for book in ("swing", "positional"):
        r = run_backtest(
            panel,
            strategies[book],
            200_000,
            CostModel(),
            SizingConfig(),
            universe.sector_of,
            days[260],
            days[-1],
        )
        assert len(r.equity) == N - 260
        closed = sum(t.net_pnl for t in r.trades)
        # Cash identity: equity = capital + closed P&L + unrealised on open positions.
        if r.open_positions == 0:
            assert abs(r.equity.iloc[-1] - (200_000 + closed)) < 1
        assert r.metrics["trades"] == len(r.trades)
        assert "benchmark_cagr" in r.metrics
        for t in r.trades:
            assert t.entry_date >= days[261] and t.exit_date >= t.entry_date
        html = render_html(r)
        assert "<svg" in html and r.variant in html
        path = write_report(r, tmp_path)
        assert path.exists() and (tmp_path / f"{path.stem}-trades.csv").exists()
    assert any(
        len(
            run_backtest(
                panel,
                strategies["swing"],
                200_000,
                CostModel(),
                SizingConfig(),
                universe.sector_of,
                days[260],
                days[-1],
            ).trades
        )
        > 0
        for _ in [0]
    )


def test_metrics_basic():
    eq = pd.Series(
        [100.0, 110.0, 99.0, 120.0], index=["2025-01-01", "2025-04-01", "2025-07-01", "2026-01-01"]
    )
    assert abs(max_drawdown(eq) - (99 / 110 - 1)) < 1e-12
    m = compute_metrics([], eq)
    assert abs(m["cagr"] - 0.2) < 0.01 and m["trades"] == 0
    assert any(line.startswith("CAGR") for line in metrics_lines(m))
