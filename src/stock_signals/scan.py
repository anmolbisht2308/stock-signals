"""Phase 2 scan: run both rule books on one date with an empty portfolio."""

from dataclasses import dataclass, field

from stock_signals.messages import signal_text
from stock_signals.models import Signal
from stock_signals.panel import Panel, Universe
from stock_signals.regime import Regime, regime_on
from stock_signals.sizing import SizingConfig, select_signals
from stock_signals.strategies import Strategy
from stock_signals.strategies.positional import PositionalRules
from stock_signals.strategies.swing import SwingRules


def build_strategies(universe: Universe, swing: bool = True) -> list[Strategy]:
    strategies: list[Strategy] = [PositionalRules(universe=universe.positional)]
    if swing:
        strategies.insert(0, SwingRules(universe=universe.swing))
    return strategies


@dataclass
class ScanResult:
    date: str
    regime: Regime
    week_end: bool
    signals: dict[str, list[Signal]] = field(default_factory=dict)


def scan(
    panel: Panel,
    universe: Universe,
    date: str,
    strategies: list[Strategy],
    capital: dict[str, float],
    sizing: SizingConfig,
) -> ScanResult:
    regime = regime_on(panel.regimes, date)
    week_end = date in panel.week_ends
    snap = panel.snapshot(date)
    result = ScanResult(date=date, regime=regime, week_end=week_end)
    for strat in strategies:
        cands = strat.candidates(snap, regime, week_end)
        result.signals[strat.book] = select_signals(
            cands,
            book=strat.book,
            date=date,
            version=strat.version,
            equity=capital[strat.book],
            free_cash=capital[strat.book],
            held=set(),
            sector_of=universe.sector_of,
            sector_counts={},
            open_count=0,
            max_new=strat.max_new(regime),
            cfg=sizing,
        )
    return result


def scan_text(result: ScanResult) -> str:
    lines = [f"🔎 Scan for {result.date}", result.regime.describe()]
    for book, signals in result.signals.items():
        lines.append("")
        if not signals:
            note = (
                " (entries only on the last session of the week)"
                if (book == "positional" and not result.week_end)
                else ""
            )
            lines.append(f"{book.upper()}: no signals{note}")
            continue
        lines.append(f"{book.upper()}: {len(signals)} signal(s)")
        lines += ["", *("\n\n".join(signal_text(s) for s in signals)).splitlines()]
    return "\n".join(lines)
