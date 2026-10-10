# Stock Signals — Phase Plan

Personal signal bot for Indian equities (NSE). It sends **me** (and a private Telegram group with only me in it) buy calls with entry, stop loss, targets and exit alerts, plus core/satellite basket rebalance alerts.

Personal use only. No subscribers, no paid group, no public channel. If that ever changes, read the SEBI notes in the research report first (a Research Analyst licence is needed to send calls to others).

**Strategy:**
1. Build data, backtesting and Telegram alerts using **simple rules** first.
2. Run it in **paper mode for 1–2 months**: alerts only, no real money.
3. Then add **LightGBM** and **Kronos**. Keep each one only if it improves the paper results (see the [keep rule](#keep-rule)).

**Trading styles:** both, run as two separate books with their own rules, risk budget and stats:

| Book | Holding period | Universe | Entry check | Exit check |
|---|---|---|---|---|
| **Swing** | 3–20 trading days | Nifty 200 (liquid) | Daily after close | Daily (intraday optional later) |
| **Positional** | 1–6 months | Nifty 500, min liquidity filter | Weekly (Friday close) | Daily stop check, weekly trailing update |

Work one phase at a time. Tick boxes as items are done.

---

## Phase 0 — Project setup

- [x] Python 3.11+ project (`uv` + `pyproject.toml`), `src/` layout, `ruff` + `pytest`
- [x] `docker-compose.yml` with MongoDB (local)
- [x] Config via `.env` + Pydantic settings (Mongo URI, Telegram token/chat id, risk %, capital)
- [ ] Create the Telegram bot with @BotFather; private group with only me; get the chat id
- [x] Universe files: Nifty 200 and Nifty 500 constituents (CSV, refreshed manually every 6 months) — `stock-signals universe --download`
- [x] Market calendar: taken from the Nifty 50 index bars (it has a bar on every session), so no holiday list to maintain
- [x] Logging to file (`data/logs/`) + a Telegram "job failed" message on any crash

**Done when:** `pytest` passes on an empty test, Mongo runs, and a "hello" message reaches the Telegram group.

## Phase 1 — Data pipeline

- [x] Download daily OHLCV for the universe + Nifty 50 index + India VIX (yfinance `.NS`; NSE bhavcopy fallback deferred until Yahoo proves unreliable)
- [x] 10+ years of history for backtests; incremental daily update after 16:00 IST (re-fetches the last 7 days to catch corrections; today's bar ignored before 16:00)
- [x] Adjust for splits and bonuses (Yahoo OHLC is split-adjusted; a new split triggers a full re-download of that stock; >30% unexplained jumps are flagged)
- [x] Store in MongoDB `bars` (`symbol`, `date` as `YYYY-MM-DD` IST, OHLCV, `adj_close`) — a regular collection with a unique (symbol, date) index so every re-run is an idempotent upsert
- [x] Data quality checks: missing days, zero volume, duplicates, bad OHLC, stale symbols, failed downloads → report in Telegram
- [x] Handle symbol changes and delistings (`symbols` collection with status active/removed/renamed; `stock-signals rename OLD NEW`)
- [x] Liquidity metric: 20-day median traded value per stock (`symbols.median_traded_value_20d`)

**Done when:** the daily job updates all symbols in under 10 minutes and the quality report is clean.

> Code and tests are done. Still to verify on my machine: first real backfill + a week of daily updates (Yahoo/NSE were not reachable from the build environment).

## Phase 2 — Indicators and rule-based signals

- [x] Indicator module (pure pandas, no TA-Lib install needed): SMA/EMA 20/50/200, ATR(14), RSI(14), 20/55-day highs, ROC 3/6/12-month, volume ratio
- [x] **Market regime filter:** Nifty above its 200-day average = risk-on (full signals); below = risk-off (swing signals halved, no new positional entries); breadth < 40% of stocks above their 50-day average = caution (swing halved)
- [x] **Swing rules (v1):**
  - Trend: close > 50 SMA > 200 SMA
  - Setup: pullback to 20 EMA and bounce, **or** 20-day high breakout with volume ≥ 1.5× average
  - Filters: RSI(14) < 80, ₹10 cr median daily value; ranked by 6-month relative strength; max 5 new/day (2 when not risk-on)
  - Stop loss: 1.5× ATR below entry (or below swing low, whichever is tighter but ≥ 1× ATR)
  - Targets: T1 = 1.5R (book half, move stop to entry), T2 = 3R or trailing 20 EMA close
  - Time stop: exit after 15 trading days if neither hit
- [x] **Positional rules (v1):**
  - Rank Nifty 500 by momentum score (50% 6-month + 50% 12-month return, divided by volatility), liquidity filter
  - Buy top-ranked names above their 200-day average, max 10 open positions
  - Stop loss: 2.5–3× ATR initially, then trail by weekly close below 50-day average
  - Exit also when the stock drops out of the top 100 rank
- [x] Position sizing: risk 1% of book capital per trade (qty = risk ₹ ÷ (entry − stop)); max 20% capital in one stock, max 3 per sector
- [x] Signal schema (Pydantic): id, book, symbol, created date, entry zone, stop, T1, T2, time stop, qty, reason, rule version
- [x] Unit tests for every rule on small hand-made price series; `stock-signals scan --date YYYY-MM-DD`

**Done when:** running the rules on any historical date produces a deterministic list of signals.

## Phase 3 — Backtesting engine

- [x] Own simple daily event-driven engine (`book.py`, shared with paper trading) (no vectorbt/backtrader in the core — licence and clarity)
- [x] Fills: entry at **next day's open** (never the signal day's close); gap-down below stop → exit at open
- [x] If stop and target both fall inside the same day's range → assume the stop hit first (conservative)
- [x] Indian costs: brokerage, STT 0.1% (delivery), exchange charges, SEBI fee, stamp duty 0.015% buy, GST 18% on charges, DP charge per sell; plus 0.1% slippage (configurable)
- [x] Metrics per book: CAGR, max drawdown, win rate, average R, expectancy (R per trade), profit factor, exposure, trades/month; compare vs Nifty 50 buy-and-hold
- [x] Walk-forward: tune on 2014–2021, test on 2022–2024, final check on 2025–now (only touched once) — `stock-signals walkforward [--final]`
- [x] Use point-in-time universe where possible; note survivorship bias where not (no free point-in-time constituents, so every report carries the survivorship warning)
- [x] HTML/Markdown report with equity curve, drawdown chart and trade list

**Done when:** both books have a report, and results on the untouched final period are not wildly worse than on the test period.

> Code and tests are done. To do on my machine after the Phase 1 backfill: `stock-signals walkforward`, read the reports in `reports/`, then `--final` once.

## Phase 4 — Telegram alerts and paper trading

- [x] Bot (`python-telegram-bot`), messages to my private group only; ignore commands from any other chat id
- [x] **Evening job (~16:30 IST):** update data → run rules → send new signals for tomorrow, e.g.:
  `🟢 SWING BUY TATAMOTORS | Entry 960–970 | SL 942 | T1 995 | T2 1030 | Qty 40 | Risk ₹1,000 | Reason: 20d breakout, vol 2.1x`
- [x] **Morning job (09:00 IST):** sends today's plan (buys at the open, planned exits). The paper fill itself happens in the evening run using that day's actual open from the EOD bar (skipped if it gapped above the entry zone or below the stop) — no intraday data needed
- [x] **Exit tracking (daily on close):** T1/T2/SL/time-stop/trailing-stop hits → exit messages (`🔴 SL HIT`, `🎯 T1 HIT — book half, SL to cost`)
- [x] Paper ledger in MongoDB (`paper_events`, `paper_trades`, `paper_equity`; book state in `paper_books`): every signal, fill, exit and P&L net of costs; records are never edited, only appended
- [x] Commands: `/plan`, `/open`, `/closed`, `/perf`, `/perf swing|positional`, `/why SYMBOL`, `/pause`, `/resume`, `/help`
- [x] Weekly summary (Saturday): trades, win rate, expectancy, equity vs Nifty, per book
- [x] Scheduling: `stock-signals serve` (bot + built-in schedule) or cron with `evening`/`morning`/`weekly`; holidays need no list — the evening run finds no new Nifty bar and does nothing. Missed days are replayed (up to 30 sessions)

**Done when:** a full week runs unattended with correct messages and a correct paper ledger.

> Code and tests are done (paper results match the backtest exactly on the same days). To verify on my machine: one unattended week.

## Phase 5 — Baskets (core + satellite)

- [ ] **Core basket:** Nifty 50 ETF or 15–20 large-cap quality stocks, equal weight; rebalance yearly or when any weight drifts >5 points
- [ ] **Momentum satellite:** top 20–30 by momentum score (Nifty200 Momentum 30 style), weights capped at 5–10%, rebalance every 6 months (June / December)
- [ ] Allocation config: e.g. 70% core / 30% satellite; satellite cut to half when Nifty is below its 200-day average
- [ ] Optimiser (skfolio or Riskfolio-Lib) only for weight caps and sector limits, not return forecasting
- [ ] Backtest the baskets with costs vs Nifty 50 and Nifty 500
- [ ] Telegram: `/basket`, rebalance alert with the exact buy/sell list and quantities for my capital
- [ ] Paper-track basket value daily

**Done when:** basket backtests are done and the first paper rebalance message is correct.

## Phase 6 — Paper run (8 weeks, no code changes to rules)

- [ ] Freeze rule versions (`swing-v1`, `positional-v1`, `basket-v1`); only bug fixes allowed
- [ ] Daily: check that the alerts match what actually happened in the market
- [ ] Weekly: compare paper results with what the backtest predicted for a similar period
- [ ] Keep a short journal of missed fills, bad data, confusing messages
- [ ] **Review at the end:**
  - ≥ 30 closed swing trades (positional will have fewer — judge on open P&L and rule-following)
  - Expectancy after costs > 0 R
  - Max drawdown within what the backtest showed
  - No data or fill bugs left open
- [ ] Decision: go to Phase 7, or fix rules and run another 4 weeks

**Done when:** the review is written in `docs/paper-review-1.md`.

## Phase 7 — LightGBM (ranker / filter)

- [ ] Features: the Phase 2 indicators + returns over several windows, volatility, volume, distance from highs, sector momentum, regime
- [ ] Label: did the trade reach T1 before SL (swing), or forward 20/60-day return rank (positional)
- [ ] Purged walk-forward training (gap between train and test so labels don't leak)
- [ ] Use it to **rank or filter** the rule signals, not to create new ones
- [ ] Backtest: rules vs rules + LightGBM, same costs, same period
- [ ] **Shadow mode** for 4–8 weeks: send both versions (`[RULES]` and `[RULES+LGBM]`) and track both paper ledgers
- [ ] Apply the [keep rule](#keep-rule)

## Phase 8 — Kronos (extra signal input)

- [ ] Install Kronos (MIT); start with `Kronos-small` zero-shot on CPU
- [ ] Patch: return all sampled paths, not just the mean; fix the normalisation look-ahead leak before any fine-tuning
- [ ] Features: probability of going up, expected return, spread of paths, P(T1 before SL) from the sampled paths
- [ ] Only evaluate on data **after mid-2024** (Kronos may have seen older data in pre-training)
- [ ] Optional: fine-tune on NSE daily data if zero-shot shows any signal (rented GPU)
- [ ] Compare against simple baselines (always-up, momentum, HAR volatility); add as LightGBM features
- [ ] Shadow mode 4–8 weeks, then apply the [keep rule](#keep-rule)

## Phase 9 — Optional extras (only after Phases 6–8)

- [ ] Live intraday SL/target alerts via Upstox (free) or Kite (₹500/month) WebSocket for my own account
- [ ] One-line LLM explanation per signal from news/announcements (Qwen/Gemini free tier — no personal data in prompts)
- [ ] Next.js dashboard for the ledger, equity curves and baskets
- [ ] Real money with small size (e.g. 25% of planned capital) for the first month

---

## Keep rule

A model (LightGBM, Kronos, or anything new) stays only if, compared with the version without it:

1. **Backtest** on the final untouched period: expectancy after costs is higher, and max drawdown is not worse by more than 20%.
2. **Shadow paper run** (≥ 4 weeks, ≥ 20 trades combined): expectancy is not lower than the rules-only version.
3. It does not make the system fragile (runs within the evening window, no manual steps).

If it fails, remove it and note why in `docs/model-decisions.md`.

## Rules for the codebase

- All dates are IST `YYYY-MM-DD`; timestamps stored in UTC.
- Signals and ledger entries are append-only; never edit history.
- No look-ahead: a signal on day D may only use data up to D's close; fills happen on D+1.
- Every rule and model has a version string saved with each signal.
- Costs are always included in backtest and paper P&L.
