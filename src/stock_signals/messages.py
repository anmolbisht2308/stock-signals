"""Telegram message text."""

from stock_signals.models import Signal


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
