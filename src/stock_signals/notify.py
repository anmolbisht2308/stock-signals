"""Messages to my private Telegram chat. Falls back to the log when no bot is configured."""

import logging
from typing import Protocol

import httpx

from stock_signals.config import Settings

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4000  # API limit is 4096 chars; keep a margin.


class Notifier(Protocol):
    def send(self, text: str) -> None: ...


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    parts, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            parts.append(current)
            current = ""
        current += line
    if current:
        parts.append(current)
    return parts


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, client: httpx.Client | None = None) -> None:
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id
        self.client = client or httpx.Client(timeout=20)

    def send(self, text: str) -> None:
        for part in split_message(text):
            resp = self.client.post(
                self.url,
                json={"chat_id": self.chat_id, "text": part, "disable_web_page_preview": True},
            )
            resp.raise_for_status()


class LogNotifier:
    def send(self, text: str) -> None:
        log.info("notify (telegram not configured):\n%s", text)


def make_notifier(settings: Settings) -> Notifier:
    if settings.telegram_bot_token and settings.telegram_chat_id:
        return TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
    return LogNotifier()
