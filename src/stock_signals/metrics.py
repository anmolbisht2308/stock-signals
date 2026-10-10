"""Performance metrics for a book."""

import numpy as np
import pandas as pd

from stock_signals.models import Trade


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    days = (pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).days
    if days <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (365.25 / days) - 1)


def _profit_factor(wins: np.ndarray, losses: np.ndarray) -> float:
    if losses.sum() < 0:
        return float(wins.sum() / -losses.sum())
    return float("inf") if len(wins) else 0.0


def compute_metrics(
    trades: list[Trade],
    equity: pd.Series,
    invested: pd.Series | None = None,
    benchmark: pd.Series | None = None,
) -> dict[str, float]:
    """`equity`/`invested`/`benchmark` are indexed by date string."""
    r = np.array([t.r_multiple for t in trades], dtype=float)
    pnl = np.array([t.net_pnl for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    months = max(len(equity) / 21, 1e-9)
    m: dict[str, float] = {
        "start_equity": float(equity.iloc[0]) if len(equity) else 0.0,
        "end_equity": float(equity.iloc[-1]) if len(equity) else 0.0,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0,
        "cagr": cagr(equity),
        "max_drawdown": max_drawdown(equity),
        "trades": float(len(trades)),
        "trades_per_month": len(trades) / months,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_r": float(r.mean()) if len(r) else 0.0,
        "avg_win_r": float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        "avg_loss_r": float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
        "profit_factor": _profit_factor(wins, losses),
        "avg_days_held": float(np.mean([t.days_held for t in trades])) if trades else 0.0,
        "total_costs": float(sum(t.costs for t in trades)),
    }
    m["expectancy_r"] = m["avg_r"]
    if invested is not None and len(invested):
        m["exposure"] = float((invested / equity.reindex(invested.index)).mean())
    if benchmark is not None and len(benchmark) > 1:
        b = benchmark.reindex(equity.index).dropna()
        m["benchmark_cagr"] = cagr(b)
        m["benchmark_max_drawdown"] = max_drawdown(b)
    return m


LABELS = {
    "total_return": ("Total return", "pct"),
    "cagr": ("CAGR", "pct"),
    "max_drawdown": ("Max drawdown", "pct"),
    "benchmark_cagr": ("Nifty 50 CAGR", "pct"),
    "benchmark_max_drawdown": ("Nifty 50 max drawdown", "pct"),
    "trades": ("Closed trades", "int"),
    "trades_per_month": ("Trades / month", "num"),
    "win_rate": ("Win rate", "pct"),
    "expectancy_r": ("Expectancy (R / trade)", "r"),
    "avg_win_r": ("Avg win", "r"),
    "avg_loss_r": ("Avg loss", "r"),
    "profit_factor": ("Profit factor", "num"),
    "avg_days_held": ("Avg days held", "num"),
    "exposure": ("Avg exposure", "pct"),
    "total_costs": ("Total costs", "rs"),
}


def fmt(key: str, value: float) -> str:
    kind = LABELS.get(key, (key, "num"))[1]
    if kind == "pct":
        return f"{value:+.1%}" if key != "win_rate" and key != "exposure" else f"{value:.0%}"
    if kind == "int":
        return f"{value:.0f}"
    if kind == "r":
        return f"{value:+.2f}R"
    if kind == "rs":
        return f"₹{value:,.0f}"
    return "∞" if value == float("inf") else f"{value:.2f}"


def metrics_lines(m: dict[str, float]) -> list[str]:
    return [f"{LABELS[k][0]}: {fmt(k, m[k])}" for k in LABELS if k in m]
