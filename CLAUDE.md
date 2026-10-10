# CLAUDE.md

Personal NSE stock signal bot: rule-based swing + positional signals, backtests, Telegram alerts, paper trading, core/satellite baskets. Later: LightGBM and Kronos, kept only if they pass the keep rule.

The roadmap is `docs/PLAN.md`. Work on **one phase at a time** and only on what that phase lists. Tick checkboxes when done.

## Stack

- Python 3.11+ (`src/` layout), pytest, ruff, Pydantic settings
- MongoDB (docker compose locally)
- python-telegram-bot, APScheduler/cron
- Data: yfinance `.NS` (personal use), NSE bhavcopy fallback
- Later: LightGBM, Kronos (MIT), skfolio/Riskfolio-Lib; optional Next.js dashboard

## Rules

- Personal use only. Never add subscriber, payment or public-channel features (SEBI RA licence needed for that).
- Dates are IST `YYYY-MM-DD`; timestamps UTC.
- No look-ahead: signals on day D use data up to D's close; paper fills at D+1 open.
- If stop and target are both inside one day's range, assume the stop hit first.
- Always include Indian costs (STT, stamp, exchange, SEBI, GST, DP) and slippage.
- Signals and the paper ledger are append-only; every signal stores its rule/model version.
- Telegram bot answers only the configured chat id. Secrets live in `.env`, never committed.
- Write tests with each rule; rule tests use small hand-made price series.
