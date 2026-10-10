from stock_signals.bot import build_application
from stock_signals.config import Settings
from stock_signals.jobs import App


def test_application_has_commands_and_schedule(db, tmp_path):
    settings = Settings(
        telegram_bot_token="123456:TEST", telegram_chat_id="-1001", data_dir=tmp_path
    )
    application = build_application(App.create(settings, db=db))
    commands = {c for h in application.handlers[0] for c in h.commands}
    assert {"plan", "open", "closed", "perf", "why", "pause", "resume", "help"} <= commands
    assert {j.name for j in application.job_queue.jobs()} == {"evening", "morning", "weekly"}


def test_requires_token(db, tmp_path):
    import pytest

    with pytest.raises(RuntimeError):
        build_application(App.create(Settings(data_dir=tmp_path), db=db))
