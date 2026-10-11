import pytest
from fastapi.testclient import TestClient

from stock_signals.api import admin_token, create_api
from stock_signals.backtest import run_backtest
from stock_signals.config import Settings
from stock_signals.costs import CostModel
from stock_signals.jobs import App, evening
from stock_signals.panel import load_panel
from stock_signals.report import write_report
from stock_signals.scan import build_strategies
from stock_signals.sizing import SizingConfig
from tests.test_paper import seed

PASSWORD = "s3cret"


@pytest.fixture
def client(db, tmp_path):
    days = seed(db)
    settings = Settings(
        data_dir=tmp_path / "data",
        admin_password=PASSWORD,
        telegram_bot_token="123:SECRET",
        mongo_uri="mongodb://user:pw@host",
    )
    app = App.create(settings, db=db)
    evening(app, update=False)
    universe = app.universe()
    panel = load_panel(app.bars, universe, end=days[-1], start=days[0])
    strat = build_strategies(universe)[1]
    r = run_backtest(
        panel, strat, 300_000, CostModel(), SizingConfig(), universe.sector_of, days[260], days[-1]
    )
    write_report(r, tmp_path / "reports")
    c = TestClient(create_api(app))
    c.headers["Authorization"] = f"Bearer {admin_token(PASSWORD)}"
    c.days = days  # type: ignore[attr-defined]
    return c


def test_login_and_auth(client):
    anon = TestClient(client.app)
    assert anon.get("/api/dashboard").status_code == 401
    assert anon.post("/api/login", json={"password": "nope"}).status_code == 401
    token = anon.post("/api/login", json={"password": PASSWORD}).json()["token"]
    assert anon.get("/api/health", headers={"Authorization": f"Bearer {token}"}).json()["ok"]


def test_no_password_configured(db, tmp_path):
    c = TestClient(create_api(App.create(Settings(data_dir=tmp_path), db=db)))
    assert c.post("/api/login", json={"password": "x"}).status_code == 503
    assert c.get("/api/health").status_code == 503


def test_dashboard(client):
    d = client.get("/api/dashboard").json()
    assert d["latest_session"] == client.days[-1]
    assert d["market"]["mode"] in {"risk_on", "caution", "risk_off"}
    assert [b["variant"] for b in d["books"]] == ["positional-v1"]  # swing off by default
    assert {b["name"] for b in d["baskets"]} == {"core", "satellite"}
    assert set(d["plan"]) == {"buys", "sells"} and d["swing_enabled"] is False


def test_positions_performance_baskets(client):
    p = client.get("/api/positions").json()
    assert set(p) == {"open", "closed", "plan"}
    perf = client.get("/api/performance").json()["books"]
    assert perf[0]["variant"] == "positional-v1" and perf[0]["series"]
    b = client.get("/api/baskets").json()
    assert b["capital"] == {"core": 350_000, "satellite": 150_000}
    assert all("holdings_detail" in x for x in b["baskets"])


def test_backtests_and_report_file(client):
    reports = client.get("/api/backtests").json()["reports"]
    assert len(reports) == 1
    rep = reports[0]
    assert rep["summary"]["variant"] == "positional-v1" and "cagr" in rep["summary"]["metrics"]
    html = client.get(f"/api/backtests/{rep['name']}/report")
    assert html.status_code == 200 and "<svg" in html.text
    assert client.get("/api/backtests/..%2F..%2Fetc%2Fpasswd/report").status_code == 404


def test_data_health_stock_settings_journal(client):
    h = client.get("/api/data-health").json()
    assert h["bars"] > 0 and h["symbols"] >= 12 and len(h["rows"]) == h["symbols"]
    s = client.get("/api/stocks/s00").json()
    assert s["symbol"]["symbol"] == "S00" and s["bars"][-1]["date"] == client.days[-1]
    assert {"sma50", "rsi14", "atr14"} <= set(s["bars"][-1])
    assert client.get("/api/stocks/NOPE").status_code == 404
    settings = client.get("/api/settings").json()
    assert settings["swing_enabled"] is False
    text = str(settings) + str(client.get("/api/dashboard").json())
    assert "SECRET" not in text and "pw@host" not in text and PASSWORD not in text
    assert client.get("/api/journal").json() == {"entries": []}
