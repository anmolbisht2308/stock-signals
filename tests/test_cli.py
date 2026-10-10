from stock_signals import cli


class Capture:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)


def test_telegram_test_sends_hello(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    cli.get_settings.cache_clear()
    n = Capture()
    assert cli.main(["telegram-test"], notifier=n) == 0
    assert n.messages == ["👋 Hello from stock-signals"]


def test_failure_is_notified(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    cli.get_settings.cache_clear()

    def boom(settings, args):
        raise RuntimeError("no mongo")

    monkeypatch.setitem(cli.COMMANDS, "update", boom)
    n = Capture()
    assert cli.main(["update"], notifier=n) == 1
    assert n.messages == ["❌ update failed: RuntimeError: no mongo"]
