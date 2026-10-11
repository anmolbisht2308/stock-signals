"""Read-only JSON API for the admin panel (Phase 10.1).

Runs on 127.0.0.1 only. Every endpoint except /api/login needs the bearer token that
/api/login returns for the ADMIN_PASSWORD in .env. Secrets (bot token, Mongo URI) are
never returned.
"""

import hashlib
import hmac
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stock_signals import dates
from stock_signals.indicators import add_indicators
from stock_signals.jobs import App
from stock_signals.jsonsafe import clean
from stock_signals.metrics import compute_metrics
from stock_signals.panel import load_panel
from stock_signals.regime import regime_on
from stock_signals.universe import MARKET_SYMBOLS

# Settings that are safe to show in the panel (no secrets).
PUBLIC_SETTINGS = [
    "swing_capital",
    "positional_capital",
    "risk_per_trade_pct",
    "max_position_pct",
    "max_per_sector",
    "max_positions",
    "slippage_pct",
    "brokerage_per_order",
    "swing_enabled",
    "basket_capital",
    "core_pct",
    "core_mode",
    "shadow_models",
    "history_years",
]


def admin_token(password: str) -> str:
    return hmac.new(password.encode(), b"stock-signals-admin", hashlib.sha256).hexdigest()


class Login(BaseModel):
    password: str


class TTLCache:
    """Tiny cache so the dashboard doesn't rebuild the 600-day indicator panel per request."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.items: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, make):  # noqa: ANN001 - any zero-arg callable
        hit = self.items.get(key)
        if hit and time.monotonic() - hit[0] < self.seconds:
            return hit[1]
        value = make()
        self.items[key] = (time.monotonic(), value)
        return value


def create_api(app: App) -> FastAPI:
    settings = app.settings
    api = FastAPI(title="stock-signals admin API", version="1")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    cache = TTLCache(seconds=300)
    reports_dir = settings.data_dir.parent / "reports"

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        if not settings.admin_password:
            raise HTTPException(503, "Set ADMIN_PASSWORD in .env to use the admin panel")
        expected = f"Bearer {admin_token(settings.admin_password)}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(401, "Not logged in")

    auth = [Depends(require_admin)]

    @api.post("/api/login")
    def login(body: Login) -> dict:
        if not settings.admin_password:
            raise HTTPException(503, "Set ADMIN_PASSWORD in .env to use the admin panel")
        if not hmac.compare_digest(body.password, settings.admin_password):
            raise HTTPException(401, "Wrong password")
        return {"token": admin_token(settings.admin_password)}

    @api.get("/api/health", dependencies=auth)
    def health() -> dict:
        return {"ok": True, "latest_session": app.latest_session()}

    def market() -> dict | None:
        latest = app.latest_session()
        if not latest:
            return None

        def build() -> dict:
            panel = load_panel(app.bars, app.universe(), end=latest)
            r = regime_on(panel.regimes, latest)
            return {
                "date": r.date,
                "mode": r.mode,
                "nifty_close": r.nifty_close,
                "nifty_sma200": r.nifty_sma200,
                "nifty_vs_sma200": r.nifty_close / r.nifty_sma200 - 1 if r.nifty_sma200 else None,
                "breadth50": r.breadth50,
                "week_end": latest in panel.week_ends,
            }

        return cache.get(f"market:{latest}", build)

    def book_summary(state) -> dict:  # noqa: ANN001 - BookState
        equity = state.equity()
        return {
            "variant": state.variant,
            "book": state.book,
            "capital": state.capital,
            "cash": state.cash,
            "equity": equity,
            "return": equity / state.capital - 1 if state.capital else 0.0,
            "open_positions": len(state.positions),
            "pending_signals": len(state.pending),
            "last_date": state.last_date,
        }

    def basket_summary(state) -> dict:  # noqa: ANN001 - BasketState
        value = state.value()
        return {
            "name": state.name,
            "capital": state.capital,
            "value": value,
            "return": value / state.capital - 1 if state.capital else 0.0,
            "invested": 1 - state.cash / value if value else 0.0,
            "invest_fraction": state.invest_fraction,
            "holdings": len(state.holdings),
            "last_selection": state.last_selection,
            "pending_orders": len(state.pending),
            "last_date": state.last_date,
        }

    def plan(states) -> dict:  # noqa: ANN001
        buys, sells = [], []
        for st in states:
            for s in st.pending:
                buys.append({"variant": st.variant, **s.model_dump()})
            for p in st.positions:
                if p.pending_exit:
                    sells.append(
                        {
                            "variant": st.variant,
                            "symbol": p.symbol,
                            "qty": p.qty,
                            "reason": p.pending_exit,
                        }
                    )
        return {"buys": buys, "sells": sells}

    @api.get("/api/dashboard", dependencies=auth)
    def dashboard() -> dict:
        states = app.paper.states()
        baskets = app.baskets().all_states()
        last_run = app.db["data_runs"].find_one({}, {"_id": 0}, sort=[("finished_at", -1)])
        return clean(
            {
                "latest_session": app.latest_session(),
                "market": market(),
                "books": [book_summary(s) for s in states],
                "baskets": [basket_summary(b) for b in baskets],
                "plan": plan(states),
                "last_data_run": last_run,
                "paused": app.paper.is_paused(),
                "swing_enabled": settings.swing_enabled,
                "schedule": {
                    "morning": "09:00 IST Mon–Fri",
                    "evening": "16:30 IST Mon–Fri",
                    "weekly": "10:00 IST Saturday",
                },
            }
        )

    @api.get("/api/positions", dependencies=auth)
    def positions(limit: int = 100) -> dict:
        states = app.paper.states()
        open_positions = []
        for st in states:
            for p in st.positions:
                pnl = (p.last_close - p.entry_price) * p.qty
                open_positions.append(
                    {
                        "variant": st.variant,
                        **p.model_dump(exclude={"legs"}),
                        "unrealised_pnl": pnl,
                        "unrealised_pct": p.last_close / p.entry_price - 1 if p.entry_price else 0,
                    }
                )
        trades = app.paper.trade_list()[-limit:][::-1]
        return clean(
            {
                "open": open_positions,
                "closed": [t.model_dump() for t in trades],
                "plan": plan(states),
            }
        )

    @api.get("/api/performance", dependencies=auth)
    def performance() -> dict:
        out = []
        for st in app.paper.states():
            eq = app.paper.equity_frame(st.variant)
            trades = app.paper.trade_list(variant=st.variant)
            item: dict = {"variant": st.variant, "book": st.book, "trades": len(trades)}
            if not eq.empty:
                item["metrics"] = compute_metrics(trades, eq["equity"], eq["invested"])
                nifty = eq["nifty"]
                item["series"] = [
                    {
                        "date": d,
                        "equity": row.equity,
                        "nifty": row.nifty / nifty.iloc[0] * st.capital if nifty.iloc[0] else None,
                    }
                    for d, row in eq.iterrows()
                ]
            out.append(item)
        return clean({"books": out})

    @api.get("/api/baskets", dependencies=auth)
    def baskets() -> dict:
        paper = app.baskets()
        out = []
        for st in paper.all_states():
            weights = st.weights()
            names = sorted(set(weights) | set(st.targets), key=lambda s: -weights.get(s, 0.0))
            holdings = [
                {
                    "symbol": s,
                    "qty": st.holdings.get(s, 0),
                    "price": st.prices.get(s),
                    "weight": weights.get(s, 0.0),
                    "target": st.targets.get(s, 0.0) * st.invest_fraction,
                    "sector": app.universe().sector_of.get(s, ""),
                }
                for s in names
            ]
            history = list(paper.equity.find({"name": st.name}, {"_id": 0}).sort("date", 1))
            events = list(
                paper.events.find({"name": st.name}, {"_id": 0, "created_at": 0})
                .sort("date", -1)
                .limit(20)
            )
            out.append(
                {
                    **basket_summary(st),
                    "holdings_detail": holdings,
                    "pending": [o.model_dump() for o in st.pending],
                    "history": history,
                    "events": events,
                }
            )
        return clean({"baskets": out, "capital": app.basket_capital()})

    @api.get("/api/backtests", dependencies=auth)
    def backtests() -> dict:
        items = []
        if reports_dir.exists():
            for html in sorted(reports_dir.glob("backtest-*.html"), reverse=True):
                summary_path = html.with_name(html.stem + "-summary.json")
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
                items.append(
                    {
                        "name": html.stem,
                        "summary": summary,
                        "modified": datetime.fromtimestamp(html.stat().st_mtime, UTC),
                    }
                )
        return clean({"reports": items})

    @api.get("/api/backtests/{name}/report", dependencies=auth)
    def backtest_report(name: str) -> FileResponse:
        path = (reports_dir / f"{name}.html").resolve()
        if path.parent != reports_dir.resolve() or not path.exists():
            raise HTTPException(404, "Report not found")
        return FileResponse(path, media_type="text/html")

    @api.get("/api/data-health", dependencies=auth)
    def data_health() -> dict:
        latest = app.latest_session()
        symbols = list(app.symbols.col.find({}, {"_id": 0}))
        stale_cutoff = None
        if latest:
            stale_cutoff = dates.to_str(dates.parse(latest) - timedelta(days=7))
        rows = []
        for d in symbols:
            last = d.get("last_bar_date")
            rows.append(
                {
                    "symbol": d["symbol"],
                    "name": d.get("name", ""),
                    "kind": d.get("kind"),
                    "status": d.get("status"),
                    "industry": d.get("industry", ""),
                    "indices": d.get("indices", []),
                    "last_bar_date": last,
                    "median_traded_value_20d": d.get("median_traded_value_20d"),
                    "stale": bool(stale_cutoff and last and last < stale_cutoff),
                }
            )
        runs = list(app.db["data_runs"].find({}, {"_id": 0}).sort("finished_at", -1).limit(10))
        return clean(
            {
                "latest_session": latest,
                "bars": app.bars.col.estimated_document_count(),
                "symbols": len(symbols),
                "active": sum(1 for r in rows if r["status"] == "active"),
                "stale": sum(1 for r in rows if r["stale"]),
                "runs": runs,
                "rows": rows,
            }
        )

    @api.get("/api/stocks/{symbol}", dependencies=auth)
    def stock(symbol: str, days: int = 400) -> dict:
        symbol = symbol.upper()
        latest = app.latest_session()
        if not latest:
            raise HTTPException(404, "No data yet")
        start = dates.to_str(dates.parse(latest) - timedelta(days=days + 320))
        bars = app.bars.load(symbol, start=start)
        if bars.empty:
            raise HTTPException(404, f"No bars for {symbol}")
        ind = add_indicators(bars)
        ind = ind[ind["date"] >= dates.to_str(dates.parse(latest) - timedelta(days=days))]
        cols = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma20",
            "sma50",
            "sma200",
            "ema20",
            "rsi14",
            "atr14",
            "roc126",
            "roc252",
        ]
        doc = app.symbols.get(symbol) or {
            "symbol": symbol,
            "kind": "market" if symbol in MARKET_SYMBOLS else None,
        }
        signals = list(
            app.paper.events.find({"symbol": symbol, "kind": "signal"}, {"_id": 0, "created_at": 0})
            .sort("date", -1)
            .limit(20)
        )
        trades = [t.model_dump() for t in app.paper.trade_list() if t.symbol == symbol]
        return clean(
            {
                "symbol": doc,
                "bars": ind[cols].to_dict("records"),
                "signals": signals,
                "trades": trades[::-1],
            }
        )

    @api.get("/api/settings", dependencies=auth)
    def get_settings_view() -> dict:
        return clean({k: getattr(settings, k) for k in PUBLIC_SETTINGS})

    @api.get("/api/journal", dependencies=auth)
    def journal() -> dict:
        return clean(
            {
                "entries": list(
                    app.db["journal"].find({}, {"_id": 0}).sort("created_at", -1).limit(200)
                )
            }
        )

    return api
