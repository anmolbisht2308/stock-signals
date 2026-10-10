"""Command line entry point: `stock-signals <command>`."""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

from stock_signals import dates, jobs
from stock_signals.backtest import PERIODS, run_backtest
from stock_signals.bars import BarStore
from stock_signals.config import Settings, get_settings
from stock_signals.costs import CostModel
from stock_signals.db import connect
from stock_signals.metrics import metrics_lines
from stock_signals.notify import Notifier, make_notifier
from stock_signals.panel import Universe, load_panel
from stock_signals.pipeline import Pipeline
from stock_signals.report import write_report
from stock_signals.scan import build_strategies, scan, scan_text
from stock_signals.sources.yahoo import YahooSource
from stock_signals.universe import (
    CALENDAR_SYMBOL,
    INDEX_URLS,
    SymbolStore,
    download_universe,
    read_universe,
    universe_path,
)

log = logging.getLogger("stock_signals")


def setup_logging(data_dir: Path) -> None:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "stock-signals.log")],
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def make_pipeline(settings: Settings) -> Pipeline:
    db = connect(settings)
    pipeline = Pipeline(
        BarStore(db),
        SymbolStore(db),
        YahooSource(),
        history_years=settings.history_years,
        batch_size=settings.batch_size,
    )
    pipeline.setup()
    return pipeline


def cmd_universe(settings: Settings, args: argparse.Namespace) -> str:
    pipeline = make_pipeline(settings)
    lines = ["📋 Universe"]
    for index in INDEX_URLS:
        path = universe_path(settings.data_dir, index)
        if args.download:
            download_universe(settings.data_dir, index)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run with --download or save the CSV from niftyindices.com"
            )
        counts = pipeline.symbols.sync_index(index, read_universe(path))
        lines.append(
            f"{index}: {counts['members']} stocks (+{counts['added']} / -{counts['removed']})"
        )
    return "\n".join(lines)


def cmd_rename(settings: Settings, args: argparse.Namespace) -> str:
    pipeline = make_pipeline(settings)
    pipeline.symbols.rename(args.old, args.new)
    pipeline.bars.delete_symbol(args.old)
    return f"🔤 {args.old} → {args.new}; run `update` to backfill {args.new}"


def cmd_backfill(settings: Settings, args: argparse.Namespace) -> str:
    return make_pipeline(settings).run(full=True, only=args.symbols).summary()


def cmd_update(settings: Settings, args: argparse.Namespace) -> str:
    return make_pipeline(settings).run(full=False, only=args.symbols).summary()


def latest_session(store: BarStore) -> str:
    last = store.last_dates().get(CALENDAR_SYMBOL)
    if not last:
        raise RuntimeError("No data yet: run `backfill` first")
    return last


def cmd_scan(settings: Settings, args: argparse.Namespace) -> str:
    db = connect(settings)
    store, symbols = BarStore(db), SymbolStore(db)
    universe = Universe.from_store(symbols)
    date = args.date or latest_session(store)
    panel = load_panel(store, universe, end=date)
    if date not in panel.calendar:
        raise ValueError(f"{date} is not a trading session in the stored data")
    result = scan(
        panel, universe, date, build_strategies(universe), settings.capital(), settings.sizing()
    )
    return scan_text(result)


def cost_model(settings: Settings) -> CostModel:
    return CostModel(
        brokerage_per_order=settings.brokerage_per_order, slippage_pct=settings.slippage_pct
    )


def _backtest(settings: Settings, books: list[str], periods: list[tuple[str, str, str]]) -> str:
    """periods: (label, start, end). Loads history once with a warm-up before the first start."""
    db = connect(settings)
    store, symbols = BarStore(db), SymbolStore(db)
    universe = Universe.from_store(symbols)
    end = min(max(p[2] for p in periods), latest_session(store))
    first = min(p[1] for p in periods)
    warmup = dates.to_str(dates.parse(first) - timedelta(days=600))
    panel = load_panel(store, universe, end=end, start=warmup)
    snapshots = panel.by_date()
    strategies = {s.book: s for s in build_strategies(universe)}
    out_dir = settings.data_dir.parent / "reports"
    lines = []
    for label, start, stop in periods:
        for book in books:
            r = run_backtest(
                panel,
                strategies[book],
                settings.capital()[book],
                cost_model(settings),
                settings.sizing(),
                universe.sector_of,
                start,
                min(stop, end),
                snapshots,
            )
            path = write_report(r, out_dir)
            lines += [
                f"📊 {label}: {r.variant} {r.start} → {r.end}",
                *metrics_lines(r.metrics),
                f"Report: {path}",
                "",
            ]
    return "\n".join(lines).strip()


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> str:
    books = ["swing", "positional"] if args.book == "all" else [args.book]
    end = args.end or "2099-12-31"
    return _backtest(settings, books, [("backtest", args.start, end)])


def cmd_walkforward(settings: Settings, args: argparse.Namespace) -> str:
    books = ["swing", "positional"] if args.book == "all" else [args.book]
    labels = ["tune", "test"] + (["final"] if args.final else [])
    return _backtest(settings, books, [(k, *PERIODS[k]) for k in labels])


def cmd_basket_backtest(settings: Settings, args: argparse.Namespace) -> str:
    from stock_signals.basket_run import build_baskets, run_basket_backtest, split_capital

    db = connect(settings)
    store, symbols = BarStore(db), SymbolStore(db)
    universe = Universe.from_store(symbols)
    end = min(args.end or "2099-12-31", latest_session(store))
    warmup = dates.to_str(dates.parse(args.start) - timedelta(days=600))
    panel = load_panel(store, universe, end=end, start=warmup)
    result = run_basket_backtest(
        panel,
        build_baskets(universe, settings.core_mode),
        split_capital(settings.basket_capital, settings.core_pct),
        cost_model(settings),
        universe.sector_of,
        args.start,
        end,
    )
    out_dir = settings.data_dir.parent / "reports"
    lines = []
    for r in [*result.sleeves.values(), result.combined]:
        path = write_report(r, out_dir)
        lines += [
            f"🧺 {r.variant} {r.start} → {r.end}",
            *metrics_lines(r.metrics),
            f"Report: {path}",
            "",
        ]
    return "\n".join(lines).strip()


def cmd_evening(settings: Settings, args: argparse.Namespace) -> str:
    return jobs.evening(jobs.App.create(settings), update=not args.no_update)


def cmd_morning(settings: Settings, args: argparse.Namespace) -> str:
    return jobs.morning(jobs.App.create(settings))


def cmd_weekly(settings: Settings, args: argparse.Namespace) -> str:
    return jobs.weekly(jobs.App.create(settings))


def cmd_serve(settings: Settings, args: argparse.Namespace) -> str:
    from stock_signals.bot import serve

    serve(jobs.App.create(settings))
    return "👋 Bot stopped"


def cmd_telegram_test(settings: Settings, args: argparse.Namespace) -> str:
    return "👋 Hello from stock-signals"


COMMANDS = {
    "universe": cmd_universe,
    "rename": cmd_rename,
    "backfill": cmd_backfill,
    "update": cmd_update,
    "scan": cmd_scan,
    "backtest": cmd_backtest,
    "walkforward": cmd_walkforward,
    "basket-backtest": cmd_basket_backtest,
    "evening": cmd_evening,
    "morning": cmd_morning,
    "weekly": cmd_weekly,
    "serve": cmd_serve,
    "telegram-test": cmd_telegram_test,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stock-signals")
    sub = p.add_subparsers(dest="command", required=True)

    u = sub.add_parser("universe", help="load Nifty 200/500 constituents into `symbols`")
    u.add_argument("--download", action="store_true", help="fetch CSVs from niftyindices.com")

    r = sub.add_parser("rename", help="record an NSE symbol change")
    r.add_argument("old")
    r.add_argument("new")

    for name, text in (("backfill", "download full history"), ("update", "fetch new bars")):
        c = sub.add_parser(name, help=text)
        c.add_argument("--symbols", nargs="+", help="only these symbols (default: universe)")

    sc = sub.add_parser("scan", help="run the rules on one date (empty portfolio)")
    sc.add_argument("--date", help="YYYY-MM-DD (default: latest session)")

    bt = sub.add_parser("backtest", help="backtest a book over a date range")
    bt.add_argument("--book", choices=["swing", "positional", "all"], default="all")
    bt.add_argument("--start", default="2016-01-01")
    bt.add_argument("--end")

    wf = sub.add_parser("walkforward", help="tune 2014–21, test 2022–24 (+ final 2025–now)")
    wf.add_argument("--book", choices=["swing", "positional", "all"], default="all")
    wf.add_argument(
        "--final", action="store_true", help="also run the untouched final period (do this once)"
    )

    bb = sub.add_parser("basket-backtest", help="backtest the core + satellite baskets")
    bb.add_argument("--start", default="2016-01-01")
    bb.add_argument("--end")

    ev = sub.add_parser("evening", help="update data, run paper trading, send signals")
    ev.add_argument("--no-update", action="store_true", help="skip the data download")
    sub.add_parser("morning", help="send today's plan")
    sub.add_parser("weekly", help="send the weekly summary")
    sub.add_parser("serve", help="run the Telegram bot with the built-in schedule")

    sub.add_parser("telegram-test", help="send a hello message")
    return p


def main(argv: list[str] | None = None, notifier: Notifier | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(settings.data_dir)
    notifier = notifier or make_notifier(settings)
    try:
        message = COMMANDS[args.command](settings, args)
    except Exception as exc:
        log.exception("%s failed", args.command)
        try:
            notifier.send(f"❌ {args.command} failed: {type(exc).__name__}: {exc}")
        except Exception:
            log.exception("could not send failure notice")
        return 1
    print(message)
    notifier.send(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
