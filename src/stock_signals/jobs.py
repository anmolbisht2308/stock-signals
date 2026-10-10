"""Scheduled jobs shared by the CLI (cron) and the bot's built-in scheduler."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from pymongo.database import Database

from stock_signals import dates
from stock_signals.bars import BarStore
from stock_signals.basket_run import BasketPaper, build_baskets, split_capital
from stock_signals.config import Settings
from stock_signals.costs import CostModel
from stock_signals.db import connect
from stock_signals.indicators import add_indicators
from stock_signals.panel import Universe, load_panel
from stock_signals.paper import PaperStore, PaperTrader
from stock_signals.pipeline import Pipeline
from stock_signals.scan import build_strategies
from stock_signals.sources.yahoo import YahooSource
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

    def trader(self, universe: Universe | None = None) -> PaperTrader:
        universe = universe or self.universe()
        return PaperTrader(
            self.paper,
            self.bars,
            universe,
            build_strategies(universe),
            self.settings.capital(),
            self.costs(),
            self.settings.sizing(),
        )


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
    parts.append(app.trader(universe).run(latest, panel).text())
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
