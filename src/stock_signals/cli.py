"""Command line entry point: `stock-signals <command>`."""

import argparse
import logging
import sys
from pathlib import Path

from stock_signals.bars import BarStore
from stock_signals.config import Settings, get_settings
from stock_signals.db import connect
from stock_signals.notify import Notifier, make_notifier
from stock_signals.pipeline import Pipeline
from stock_signals.sources.yahoo import YahooSource
from stock_signals.universe import (
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


def cmd_telegram_test(settings: Settings, args: argparse.Namespace) -> str:
    return "👋 Hello from stock-signals"


COMMANDS = {
    "universe": cmd_universe,
    "rename": cmd_rename,
    "backfill": cmd_backfill,
    "update": cmd_update,
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
