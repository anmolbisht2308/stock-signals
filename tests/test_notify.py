import json

import httpx

from stock_signals.notify import TelegramNotifier, split_message


def test_split_message_respects_limit():
    text = "\n".join(["x" * 30] * 10)
    parts = split_message(text, limit=100)
    assert all(len(p) <= 100 for p in parts) and "".join(parts) == text
    assert split_message("y" * 250, limit=100) == ["y" * 100, "y" * 100, "y" * 50]


def test_telegram_posts_to_chat():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    TelegramNotifier("TOKEN", "123", client=client).send("hi")
    assert sent == [{"chat_id": "123", "text": "hi", "disable_web_page_preview": True}]
