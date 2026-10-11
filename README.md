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
uv run stock-signals universe --download   # Nifty 100 / 200 / 500 lists from niftyindices.com
uv run stock-signals backfill              # ~13 years of daily bars (first run takes a while)
uv run stock-signals update                # daily, after 16:00 IST
uv run stock-signals rename OLDSYM NEWSYM  # when NSE changes a symbol
```

If the niftyindices download is blocked, save the CSVs from the Nifty 100 / 200 / 500 index pages into `data/universe/nifty100.csv`, `nifty200.csv` and `nifty500.csv` and run `universe` without `--download`.

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

The swing book is **off** (`SWING_ENABLED=false`): it showed no edge after costs in any backtest period. Scans and paper trading run the positional book and the baskets; `backtest --book swing` still works for experiments.

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

## Baskets (Phase 5)

- **Core** (default 70% of `BASKET_CAPITAL`): 20 lowest-volatility Nifty 100 stocks, equal weight, reselected each January, weights restored monthly if they drift > 5 points. `CORE_MODE=etf` uses NIFTYBEES instead.
- **Satellite** (30%): top 30 Nifty 200 momentum stocks, score-weighted, max 10% per stock and 30% per sector, reselected in June and December; cut to 50% invested while the market is risk-off.

```bash
uv run stock-signals basket-backtest --start 2016-01-01
```

The evening run paper-tracks both baskets and posts rebalance orders; `/basket` shows holdings and weights.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Paper review (Phase 6)

During the 8-week paper run, jot anything odd with `/note …` in Telegram. At the end:

```bash
uv run stock-signals review     # writes docs/paper-review-N.md with the go/no-go checks
```

## LightGBM (Phase 7)

The model only filters and re-ranks rule signals; it never creates new ones.

```bash
uv run stock-signals ml-evaluate            # rules vs rules+LightGBM on 2022–24 (walk-forward, purged)
uv run stock-signals ml-evaluate --final    # decide on 2025–now (keep rule), logged to docs/model-decisions.md
uv run stock-signals ml-train --book swing  # only after a KEEP: saves models/swing-lgbm.txt
```

With a saved model, the evening run adds a shadow paper book (`swing-v1+lgbm`) next to the rules-only one; messages are tagged `[variant]`. `review` then checks the paper half of the keep rule. Delete the model file to drop it.

## Kronos (Phase 8)

```bash
git clone https://github.com/shiyu-coder/Kronos ~/Kronos
uv pip install torch einops huggingface_hub safetensors
echo "KRONOS_PATH=~/Kronos" >> .env

uv run stock-signals kronos-features        # sample paths for every swing candidate since 2024-07-01 (slow on CPU)
uv run stock-signals kronos-report          # Kronos vs always-up and momentum baselines
uv run stock-signals ml-evaluate --book swing --kronos
uv run stock-signals ml-train --book swing --kronos   # only after a KEEP
```

Kronos is only judged on data after mid-2024 (its pre-training may include earlier prices). If it doesn't beat the baselines, don't use it.

## Admin panel API (Phase 10)

The panel lives in a separate repo: [stock-signals-frontend](https://github.com/anmolbisht2308/stock-signals-frontend). This repo serves its data:

```bash
echo "ADMIN_PASSWORD=choose-a-password" >> .env
uv run stock-signals api        # http://127.0.0.1:8000/docs lists every endpoint
```

The API listens on your Mac only (127.0.0.1), never returns the bot token or Mongo URI, and every endpoint needs the token from `POST /api/login`.
