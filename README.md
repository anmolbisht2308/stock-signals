# Stock Signals

Personal swing + positional signal bot for NSE stocks with Telegram alerts, paper trading and core/satellite baskets. Runs on localhost; everything used is free.

Personal use only — signals go to my own Telegram, not to any subscribers.

- Phase plan: [`docs/PLAN.md`](docs/PLAN.md) — currently **Phase 1 (data)**
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

Every command posts its summary (or failure) to Telegram. Daily cron (Mon–Fri, 16:30 IST):

```cron
30 16 * * 1-5  cd /path/to/stock-signals && uv run stock-signals update >> data/logs/cron.log 2>&1
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
