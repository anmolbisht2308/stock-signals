"""Telegram bot with built-in schedule. Run `stock-signals serve` and leave it running.

Answers only the configured chat id. Schedule (IST): evening job Mon–Fri 16:30, morning
plan Mon–Fri 09:00, weekly summary Saturday 10:00.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import time

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, filters

from stock_signals import jobs
from stock_signals.dates import IST
from stock_signals.notify import split_message
from stock_signals.views import HELP, basket_text, closed_text, open_text, perf_text, why_text

log = logging.getLogger(__name__)

# python-telegram-bot day numbers: 0 = Sunday … 6 = Saturday.
WEEKDAYS = (1, 2, 3, 4, 5)
SATURDAY = (6,)
EVENING = time(16, 30, tzinfo=IST)
MORNING = time(9, 0, tzinfo=IST)
WEEKLY = time(10, 0, tzinfo=IST)


def build_application(app: jobs.App) -> Application:
    settings = app.settings
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
    chat_id = int(settings.telegram_chat_id)
    application = Application.builder().token(settings.telegram_bot_token).build()
    only_me = filters.Chat(chat_id=chat_id)

    async def send(bot, text: str) -> None:  # noqa: ANN001 - telegram.Bot
        for part in split_message(text):
            await bot.send_message(chat_id=chat_id, text=part, disable_web_page_preview=True)

    def command(name: str, make_text: Callable[[list[str]], str]) -> None:
        async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            try:
                text = await asyncio.to_thread(make_text, context.args or [])
            except Exception as exc:  # report any failure back to the chat
                log.exception("/%s failed", name)
                text = f"❌ /{name} failed: {exc}"
            for part in split_message(text):
                await update.effective_message.reply_text(part)

        application.add_handler(CommandHandler(name, handler, filters=only_me))

    def book_arg(args: list[str]) -> str | None:
        return args[0].lower() if args and args[0].lower() in ("swing", "positional") else None

    def pause(value: bool) -> str:
        app.paper.set_paused(value)
        return (
            "⏸️ Paused: no new signals, open positions still managed"
            if value
            else ("▶️ Resumed: new signals from the next evening run")
        )

    command("start", lambda a: HELP)
    command("help", lambda a: HELP)
    command("plan", lambda a: jobs.morning(app))
    command("open", lambda a: open_text(app.paper.states()))
    command("closed", lambda a: closed_text(app.paper.trade_list()))
    command("perf", lambda a: perf_text(app.paper, app.paper.states(), book=book_arg(a)))
    command(
        "why",
        lambda a: (
            why_text(app.paper, a[0], jobs.symbol_indicators(app, a[0]))
            if a
            else "Usage: /why SYMBOL"
        ),
    )
    command("basket", lambda a: basket_text(app.baskets().all_states(), app.basket_capital()))

    def note(args: list[str]) -> str:
        from stock_signals.review import Journal

        if not args:
            return "Usage: /note what happened"
        Journal(app.db).add(" ".join(args))
        return "📝 Noted for the paper review"

    command("note", note)
    command("pause", lambda a: pause(True))
    command("resume", lambda a: pause(False))

    def scheduled(name: str, fn: Callable[[], str]):
        async def job(context: ContextTypes.DEFAULT_TYPE) -> None:
            try:
                text = await asyncio.to_thread(fn)
            except Exception as exc:
                log.exception("%s job failed", name)
                text = f"❌ {name} failed: {type(exc).__name__}: {exc}"
            await send(context.bot, text)

        return job

    jq = application.job_queue
    jq.run_daily(
        scheduled("evening", lambda: jobs.evening(app)), EVENING, days=WEEKDAYS, name="evening"
    )
    jq.run_daily(
        scheduled("morning", lambda: jobs.morning(app)), MORNING, days=WEEKDAYS, name="morning"
    )
    jq.run_daily(
        scheduled("weekly", lambda: jobs.weekly(app)), WEEKLY, days=SATURDAY, name="weekly"
    )
    return application


def serve(app: jobs.App) -> None:
    application = build_application(app)
    log.info("bot running; Ctrl+C to stop")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
