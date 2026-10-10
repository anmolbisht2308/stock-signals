"""Telegram message text."""

from stock_signals.models import Position, Signal, Trade


def rs(value: float) -> str:
    """₹ amount without paise for big numbers, Indian digit grouping."""
    negative = value < 0
    n = f"{abs(round(value)):d}"
    if len(n) > 3:
        head, tail = n[:-3], n[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        n = ",".join(groups) + "," + tail
    return ("-₹" if negative else "₹") + n


def px(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:,.2f}" if value < 100 else f"{value:,.1f}"


def signal_text(s: Signal) -> str:
    targets = ""
    if s.t1:
        targets += f" | T1 {px(s.t1)}"
    if s.t2:
        targets += f" | T2 {px(s.t2)}"
    hold = f" | Time stop {s.time_stop_days}d" if s.time_stop_days else ""
    return (
        f"🟢 {s.book.upper()} BUY {s.symbol}\n"
        f"Buy at open if ≤ {px(s.entry_high)} (close {px(s.ref_price)})\n"
        f"SL {px(s.stop)}{targets}{hold}\n"
        f"Qty {s.qty} | Risk {rs(s.risk_amount)}\n"
        f"Why: {s.reason}"
    )


REASONS = {
    "stop": "🔴 SL HIT",
    "breakeven_stop": "⚪ Stopped at cost",
    "t1": "🎯 T1 HIT",
    "t2": "🎯🎯 T2 HIT",
    "time_stop": "⏱️ Time stop",
    "trail_ema20": "📉 Closed below 20 EMA",
    "trail_sma50": "📉 Weekly close below 50-DMA",
    "rank_exit": "📉 Momentum rank dropped",
}


def reason_label(reason: str) -> str:
    return REASONS.get(reason, reason)


def fill_text(p: Position) -> str:
    return (
        f"✅ {p.book.upper()} BOUGHT {p.symbol} {p.qty} @ {px(p.entry_price)} (paper)\n"
        f"SL {px(p.stop)}"
        + (f" | T1 {px(p.t1)}" if p.t1 else "")
        + (f" | T2 {px(p.t2)}" if p.t2 else "")
    )


def skip_text(s: Signal, why: str) -> str:
    return f"⏭️ {s.book.upper()} skipped {s.symbol}: {why}"


def partial_text(p: Position, price: float, qty: int, reason: str) -> str:
    return (
        f"{reason_label(reason)} {p.book.upper()} {p.symbol}: sold {qty} @ {px(price)}, "
        f"{p.qty} left"
    )


def stop_moved_text(p: Position) -> str:
    return f"🛡️ {p.book.upper()} {p.symbol}: SL moved to cost {px(p.stop)}"


def exit_planned_text(p: Position, reason: str) -> str:
    return f"🚪 {p.book.upper()} {p.symbol}: {reason_label(reason)} — sell {p.qty} at next open"


def exit_text(t: Trade, price: float) -> str:
    return (
        f"{reason_label(t.exit_reason)} {t.book.upper()} {t.symbol}: exited @ {px(price)}\n"
        f"P&L {rs(t.net_pnl)} after costs ({t.r_multiple:+.2f}R) in {t.days_held}d"
    )
