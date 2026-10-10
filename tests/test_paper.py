import pytest

from stock_signals.backtest import run_backtest
from stock_signals.bars import BarStore
from stock_signals.config import Settings
from stock_signals.costs import CostModel
from stock_signals.jobs import App, evening, morning, weekly
from stock_signals.panel import Universe, build_panel
from stock_signals.paper import PaperStore, PaperTrader
from stock_signals.scan import build_strategies
from stock_signals.sizing import SizingConfig
from stock_signals.universe import Member, SymbolStore
from stock_signals.views import closed_text, open_text, perf_text, plan_text, why_text
from tests.test_backtest import market

SYMS = {f"S{i:02d}" for i in range(12)}
UNIVERSE = Universe(SYMS, SYMS, {s: f"Sec{int(s[1:]) % 4}" for s in SYMS})
CAPITAL = {"swing": 200_000, "positional": 300_000}


@pytest.fixture(scope="module")
def panel_and_days():
    bars, days = market()
    return build_panel(bars, UNIVERSE), days


def trader(db):
    return PaperTrader(
        PaperStore(db),
        BarStore(db),
        UNIVERSE,
        build_strategies(UNIVERSE),
        CAPITAL,
        CostModel(),
        SizingConfig(),
    )


def test_first_run_then_catch_up_then_idempotent(db, panel_and_days):
    panel, days = panel_and_days
    t = trader(db)
    run = t.run(days[300], panel)
    assert [d.date for d in run.days] == [days[300]]
    st = t.store.load_state("swing-v1")
    assert st.last_date == days[300] and st.cash == 200_000

    run = t.run(days[303], panel)
    assert [d.date for d in run.days] == days[301:304]
    assert db["paper_equity"].count_documents({"variant": "swing-v1"}) == 4
    assert "📅" in run.text()

    again = t.run(days[303], panel)
    assert again.days == [] and "Nothing new" in again.text()


def test_paper_matches_backtest(db, panel_and_days):
    panel, days = panel_and_days
    t = trader(db)
    for d in days[260:400]:
        t.run(d, panel)
    paper_state = t.store.load_state("swing-v1")
    swing = build_strategies(UNIVERSE)[0]
    bt = run_backtest(
        panel, swing, 200_000, CostModel(), SizingConfig(), UNIVERSE.sector_of, days[260], days[399]
    )
    assert paper_state.equity() == pytest.approx(bt.equity.iloc[-1])
    assert len(t.store.trade_list(variant="swing-v1")) == len(bt.trades)
    assert len(bt.trades) > 0


def test_pause_stops_new_signals(db, panel_and_days):
    panel, days = panel_and_days
    t = trader(db)
    t.store.set_paused(True)
    run = t.run(days[330], panel)
    for d in days[331:360]:
        run = t.run(d, panel)
        assert all(e.kind != "signal" for _, e in run.days[0].events)
    assert "Paused" in run.text()
    t.store.set_paused(False)
    assert not t.store.is_paused()


def test_views(db, panel_and_days):
    panel, days = panel_and_days
    t = trader(db)
    for d in days[260:340]:
        t.run(d, panel)
    states = t.store.states()
    assert [s.variant for s in states] == ["positional-v1", "swing-v1"]
    assert "Today's plan" in plan_text(states)
    assert "swing-v1: cash" in open_text(states)
    trades = t.store.trade_list()
    assert ("closed trades" in closed_text(trades)) == bool(trades)
    perf = perf_text(t.store, states)
    assert "vs Nifty" in perf and "expectancy" in perf
    assert "swing-v1" in perf_text(t.store, states, book="swing")
    assert "positional-v1" not in perf_text(t.store, states, book="swing")
    assert "No signal recorded" in why_text(t.store, "NOPE", None)


def seed(db):
    bars, days = market()
    # insert_many: mongomock upserts are O(n²) and this is ~5k bars.
    db["bars"].insert_many(bars.to_dict("records"))
    symbols = SymbolStore(db)
    members = [Member(s, s, f"Sec{int(s[1:]) % 4}", "") for s in sorted(SYMS)]
    symbols.sync_index("NIFTY200", members)
    symbols.sync_index("NIFTY500", members)
    return days


def test_jobs_end_to_end(db, tmp_path):
    days = seed(db)
    app = App.create(Settings(data_dir=tmp_path), db=db)
    text = evening(app, update=False)
    assert days[-1] in text and "Market" in text
    assert "Today's plan" in morning(app)
    w = weekly(app)
    assert "Weekly summary" in w and "Since start" in w


def test_shadow_model_book(db, tmp_path):
    from stock_signals.ml.evaluate import train_final

    days = seed(db)
    settings = Settings(data_dir=tmp_path / "data")
    app = App.create(settings, db=db)
    universe = app.universe()
    from stock_signals.panel import load_panel

    panel = load_panel(app.bars, universe, end=days[-1], start=days[0])
    swing = build_strategies(universe)[0]
    train_final(panel, swing, universe.sector_of, days[0], days[-1]).save(
        settings.models_dir / "swing-lgbm.txt"
    )
    versions = [s.version for s in app.strategies(universe, panel)]
    assert versions == ["swing-v1", "positional-v1", "swing-v1+lgbm"]
    text = evening(app, update=False)
    assert days[-1] in text
    assert app.paper.load_state("swing-v1+lgbm") is not None
    settings_off = Settings(data_dir=tmp_path / "data", shadow_models=False)
    assert len(App.create(settings_off, db=db).strategies(universe, panel)) == 2
