"""Text for bot commands and scheduled summaries (no Telegram dependency, easy to test)."""

from stock_signals.book import BookState
from stock_signals.messages import px, reason_label, rs
from stock_signals.metrics import compute_metrics
from stock_signals.models import Trade
from stock_signals.paper import PaperStore


def plan_text(states: list[BookState]) -> str:
    lines = ["🗓️ Today's plan (paper)"]
    any_item = False
    for st in states:
        buys = [s for s in st.pending]
        exits = [p for p in st.positions if p.pending_exit]
        if not buys and not exits:
            continue
        any_item = True
        lines.append(f"\n{st.variant}")
        for p in exits:
            lines.append(f"🚪 SELL {p.symbol} {p.qty} at open — {reason_label(p.pending_exit)}")
        for s in buys:
            lines.append(
                f"🟢 BUY {s.symbol} {s.qty} at open if ≤ {px(s.entry_high)} | SL {px(s.stop)}"
                + (f" | T1 {px(s.t1)}" if s.t1 else "")
            )
    if not any_item:
        lines.append("Nothing to do at the open.")
    return "\n".join(lines)


def open_text(states: list[BookState]) -> str:
    lines = ["📂 Open positions (paper)"]
    total = 0
    for st in states:
        lines.append(f"\n{st.variant}: cash {rs(st.cash)}, equity {rs(st.equity())}")
        if not st.positions:
            lines.append("  none")
        for p in st.positions:
            total += 1
            pnl = (p.last_close - p.entry_price) * p.qty
            pct = p.last_close / p.entry_price - 1
            note = f" | exit at open: {reason_label(p.pending_exit)}" if p.pending_exit else ""
            targets = (f" T1 {px(p.t1)}{' ✓' if p.t1_done else ''}" if p.t1 else "") + (
                f" T2 {px(p.t2)}" if p.t2 else ""
            )
            lines.append(
                f"• {p.symbol} {p.qty} @ {px(p.entry_price)} → {px(p.last_close)} "
                f"({pct:+.1%}, {rs(pnl)}) | SL {px(p.stop)}{targets} | {p.bars_held}d{note}"
            )
    return "\n".join(lines)


def closed_text(trades: list[Trade], limit: int = 10) -> str:
    if not trades:
        return "📕 No closed trades yet"
    lines = [f"📕 Last {min(limit, len(trades))} closed trades"]
    for t in trades[-limit:][::-1]:
        lines.append(
            f"• {t.symbol} ({t.book}) {t.entry_date} → {t.exit_date}: {rs(t.net_pnl)} "
            f"({t.r_multiple:+.2f}R) {reason_label(t.exit_reason)}"
        )
    return "\n".join(lines)


def perf_text(
    store: PaperStore,
    states: list[BookState],
    book: str | None = None,
    since: str | None = None,
    title: str = "📈 Paper performance",
) -> str:
    lines = [title + (f" since {since}" if since else "")]
    for st in states:
        if book and st.book != book:
            continue
        trades = store.trade_list(variant=st.variant, since=since)
        eq = store.equity_frame(st.variant)
        if since and not eq.empty:
            eq = eq[eq.index >= since]
        lines.append(f"\n{st.variant}")
        if eq.empty:
            lines.append("  no sessions yet")
            continue
        m = compute_metrics(trades, eq["equity"], eq["invested"])
        nifty = eq["nifty"].dropna()
        nifty_ret = float(nifty.iloc[-1] / nifty.iloc[0] - 1) if len(nifty) > 1 else 0.0
        start_eq = st.capital if not since else float(eq["equity"].iloc[0])
        ret = float(eq["equity"].iloc[-1] / start_eq - 1)
        lines += [
            f"  Equity {rs(float(eq['equity'].iloc[-1]))} ({ret:+.1%}) vs Nifty {nifty_ret:+.1%}",
            f"  Max drawdown {m['max_drawdown']:.1%} | open positions {len(st.positions)}",
            f"  Closed {len(trades)} | win rate {m['win_rate']:.0%} | "
            f"expectancy {m['expectancy_r']:+.2f}R | P&L {rs(sum(t.net_pnl for t in trades))}",
        ]
    return "\n".join(lines)


def why_text(store: PaperStore, symbol: str, indicators: dict | None) -> str:
    symbol = symbol.upper()
    lines = [f"🔍 {symbol}"]
    sig = store.last_signal(symbol)
    if sig:
        d = sig.get("data", {}).get("signal", {})
        lines.append(f"Last signal {sig['date']} ({sig['book']}): {d.get('reason', '')}")
        lines.append(
            f"Entry ≤ {px(d.get('entry_high'))} | SL {px(d.get('stop'))}"
            + (f" | T1 {px(d.get('t1'))} | T2 {px(d.get('t2'))}" if d.get("t1") else "")
        )
    else:
        lines.append("No signal recorded for this symbol.")
    if indicators:
        i = indicators
        trend = "up" if i["close"] > i["sma50"] > i["sma200"] else "not up"
        lines.append(
            f"{i['date']}: close {px(i['close'])} | 50-DMA {px(i['sma50'])} | "
            f"200-DMA {px(i['sma200'])} | trend {trend} | RSI {i['rsi14']:.0f} | "
            f"ATR {px(i['atr14'])} | 6m {i['roc126']:+.0%}"
        )
    return "\n".join(lines)


def basket_text(states: list, capital: dict[str, float]) -> str:
    """`states`: BasketState list; quantities are for the configured basket capital."""
    if not states:
        return "🧺 No baskets yet — they start with the next evening run"
    lines = ["🧺 Baskets (paper)"]
    for st in states:
        value = st.value()
        ret = value / st.capital - 1 if st.capital else 0.0
        invested = 1 - st.cash / value if value else 0.0
        lines.append(
            f"\n{st.name.upper()}: {rs(value)} ({ret:+.1%}), {invested:.0%} invested, "
            f"selected {st.last_selection or '–'}"
        )
        w = st.weights()
        for sym, weight in sorted(w.items(), key=lambda kv: -kv[1]):
            target = st.targets.get(sym, 0.0) * st.invest_fraction
            lines.append(f"• {sym} {st.holdings[sym]} sh, {weight:.1%} (target {target:.1%})")
        if st.pending:
            parts = [
                f"{'BUY' if o.qty > 0 else 'SELL'} {o.symbol} {abs(o.qty)}" for o in st.pending
            ]
            lines.append("Next open: " + ", ".join(parts))
    return "\n".join(lines)


HELP = """🤖 Commands
/plan — today's buys and sells at the open
/open — open paper positions
/closed — last 10 closed trades
/perf [swing|positional] — performance vs Nifty
/why SYMBOL — last signal and current indicators
/basket — core and satellite baskets
/note TEXT — add to the paper-run journal
/pause, /resume — stop or restart new signals
/help — this list"""
