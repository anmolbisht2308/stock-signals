"""Scheduled jobs shared by the CLI (cron) and the bot's built-in scheduler."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from pymongo.database import Database

from stock_signals import dates
from stock_signals.bars import BarStore
from stock_signals.basket_run import BasketPaper, build_baskets, split_capital
from stock_signals.config import Settings
from stock_signals.costs import CostModel
from stock_signals.db import connect
from stock_signals.indicators import add_indicators
from stock_signals.ml.model import TrainedModel
from stock_signals.ml.scored import ModelScorer, ScoredStrategy
from stock_signals.panel import Panel, Universe, load_panel
from stock_signals.paper import PaperStore, PaperTrader
from stock_signals.pipeline import Pipeline
from stock_signals.scan import build_strategies
from stock_signals.sources.yahoo import YahooSource
from stock_signals.strategies import Strategy
from stock_signals.universe import CALENDAR_SYMBOL, SymbolStore
from stock_signals.views import perf_text, plan_text


@dataclass
class App:
    settings: Settings
    db: Database
    bars: BarStore
    symbols: SymbolStore
    paper: PaperStore

    @classmethod
    def create(cls, settings: Settings, db: Database | None = None) -> "App":
        db = db if db is not None else connect(settings)
        return cls(settings, db, BarStore(db), SymbolStore(db), PaperStore(db))

    def costs(self) -> CostModel:
        s = self.settings
        return CostModel(brokerage_per_order=s.brokerage_per_order, slippage_pct=s.slippage_pct)

    def latest_session(self) -> str | None:
        return self.bars.last_dates().get(CALENDAR_SYMBOL)

    def universe(self) -> Universe:
        return Universe.from_store(self.symbols)

    def basket_capital(self) -> dict[str, float]:
        return split_capital(self.settings.basket_capital, self.settings.core_pct)

    def baskets(self, universe: Universe | None = None) -> BasketPaper:
        universe = universe or self.universe()
        return BasketPaper(
            self.db,
            build_baskets(universe, self.settings.core_mode),
            self.basket_capital(),
            self.costs(),
            universe.sector_of,
        )

    def strategies(self, universe: Universe, panel: Panel | None = None) -> list[Strategy]:
        """Rules-only books, plus a shadow book per saved model (Phase 7/8)."""
        base = build_strategies(universe)
        if panel is None or not self.settings.shadow_models:
            return base
        shadows: list[Strategy] = []
        for strat in base:
            for name in model_names(self.settings.models_dir, strat.book):
                model = TrainedModel.load(self.settings.models_dir / name)
                extra = kronos_features(self.db, panel) if "kronos" in name else None
                scorer = ModelScorer(model, universe.sector_of, panel.regimes, extra)
                suffix = name.removesuffix(".txt").split("-", 2)[-1]
                shadows.append(ScoredStrategy(strat, scorer, model.threshold, suffix))
        return base + shadows

    def trader(self, universe: Universe | None = None, panel: Panel | None = None) -> PaperTrader:
        universe = universe or self.universe()
        return PaperTrader(
            self.paper,
            self.bars,
            universe,
            self.strategies(universe, panel),
            self.settings.capital(),
            self.costs(),
            self.settings.sizing(),
        )


def model_names(models_dir: Path, book: str) -> list[str]:
    """Saved models for a book: models/<book>-lgbm.txt, models/<book>-lgbm_kronos.txt …"""
    if not models_dir.exists():
        return []
    return sorted(p.name for p in models_dir.glob(f"{book}-*.txt"))


def kronos_features(db: Database, panel: Panel):  # -> pd.DataFrame | None
    from stock_signals.ml.kronos import KronosCache

    return KronosCache(db).frame(start=panel.calendar[0] if panel.calendar else None)


def update_data(app: App, now: datetime | None = None) -> str:
    pipeline = Pipeline(
        app.bars,
        app.symbols,
        YahooSource(),
        history_years=app.settings.history_years,
        batch_size=app.settings.batch_size,
    )
    pipeline.setup()
    result = pipeline.run(now=now)
    if result.failed or not result.quality.ok:
        return result.summary()
    return f"📥 Data updated to {result.end} ({result.symbols} symbols)"


def evening(app: App, now: datetime | None = None, update: bool = True) -> str:
    parts = [update_data(app, now)] if update else []
    latest = app.latest_session()
    if not latest:
        raise RuntimeError("No data yet: run `backfill` first")
    universe = app.universe()
    panel = load_panel(app.bars, universe, end=latest)
    trader = app.trader(universe, panel)
    books = [s.book for s in trader.strategies]
    multi = {b for b in books if books.count(b) > 1}
    parts.append(trader.run(latest, panel).text(multi))
    basket_lines = app.baskets(universe).run(latest, panel)
    if basket_lines:
        parts.append("\n".join(basket_lines))
    return "\n\n".join(parts)


def morning(app: App) -> str:
    return plan_text(app.paper.states())


def weekly(app: App, today: date | None = None) -> str:
    today = today or dates.now_ist().date()
    since = dates.to_str(today - timedelta(days=7))
    states = app.paper.states()
    return (
        perf_text(app.paper, states, since=since, title="🗞️ Weekly summary")
        + "\n\n"
        + perf_text(app.paper, states, title="📈 Since start")
    )


def symbol_indicators(app: App, symbol: str) -> dict | None:
    latest = app.latest_session()
    if not latest:
        return None
    start = dates.to_str(dates.parse(latest) - timedelta(days=420))
    bars = app.bars.load(symbol.upper(), start=start)
    if bars.empty:
        return None
    row = add_indicators(bars).iloc[-1]
    if row.isna()[["sma50", "sma200", "rsi14", "atr14", "roc126"]].any():
        return None
    return row.to_dict()
