# Stock Signals

Personal swing + positional signal bot for NSE stocks with Telegram alerts, paper trading and core/satellite baskets. Runs on localhost; everything used is free.

Personal use only — signals go to my own Telegram, not to any subscribers.

- Phase plan: [`docs/PLAN.md`](docs/PLAN.md)
- Research behind it: [`docs/research.md`](docs/research.md)

## Setup

```bash
# 1. MongoDB (or install MongoDB Community natively)
docker compose up -d

# 2. Python deps (https://docs.astral.sh/uv/)
uv sync

# 3. Config
cp .env.example .env
# Telegram: message @BotFather → /newbot → paste the token into TELEGRAM_BOT_TOKEN.
# Create a private group with only you + the bot, send any message, then open
# https://api.telegram.org/bot<TOKEN>/getUpdates and copy chat.id into TELEGRAM_CHAT_ID.
uv run stock-signals telegram-test
```

## Data (Phase 1)

```bash
uv run stock-signals universe --download   # Nifty 200 + 500 lists from niftyindices.com
uv run stock-signals backfill              # ~10 years of daily bars (first run takes a while)
uv run stock-signals update                # daily, after 16:00 IST
uv run stock-signals rename OLDSYM NEWSYM  # when NSE changes a symbol
```

If the niftyindices download is blocked, save the CSVs from the Nifty 200 / Nifty 500 index pages into `data/universe/nifty200.csv` and `data/universe/nifty500.csv` and run `universe` without `--download`.

Every command posts its summary (or failure) to Telegram.

## Rules, backtests (Phases 2–3)

```bash
uv run stock-signals scan                       # signals for the latest session (empty portfolio)
uv run stock-signals scan --date 2025-03-14
uv run stock-signals backtest --book swing --start 2018-01-01
uv run stock-signals walkforward                # tune 2014–21 + test 2022–24
uv run stock-signals walkforward --final        # adds 2025–now; do this once
```

HTML reports, trade lists and equity CSVs go to `reports/`.

## Paper trading on Telegram (Phase 4)

Leave this running (PC on during market days):

```bash
uv run stock-signals serve
```

It answers `/plan /open /closed /perf /why SYMBOL /pause /resume /help` (only from your chat) and runs:

| When (IST) | Job |
|---|---|
| Mon–Fri 09:00 | today's plan: buys at the open, planned exits |
| Mon–Fri 16:30 | download data → paper fills/exits for the day → new signals for tomorrow |
| Sat 10:00 | weekly summary vs Nifty |

Prefer cron? Use the same jobs without the bot:

```cron
0 9  * * 1-5  cd /path/to/stock-signals && uv run stock-signals morning
30 16 * * 1-5 cd /path/to/stock-signals && uv run stock-signals evening
0 10 * * 6    cd /path/to/stock-signals && uv run stock-signals weekly
```

If the PC was off, the next evening run replays the missed sessions (up to 30).

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
