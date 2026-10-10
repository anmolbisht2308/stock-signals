"""Data quality checks on daily bars.

The trading calendar is taken from the Nifty 50 index bars: the index has a bar
on every NSE session, so no holiday list is needed.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

# Overnight close-to-close move treated as a possible unadjusted split/bonus.
JUMP_THRESHOLD = 0.30
# Sessions without a bar before a symbol is called stale.
STALE_SESSIONS = 5


@dataclass(frozen=True)
class Issue:
    kind: str
    symbol: str
    detail: str


@dataclass
class QualityReport:
    checked: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def extend(self, other: "QualityReport") -> None:
        self.checked += other.checked
        self.issues.extend(other.issues)

    def summary(self, max_per_kind: int = 10) -> str:
        if self.ok:
            return f"✅ Data quality OK ({self.checked} symbols)"
        by_kind: dict[str, list[Issue]] = defaultdict(list)
        for i in self.issues:
            by_kind[i.kind].append(i)
        lines = [f"⚠️ Data quality: {len(self.issues)} issues across {self.checked} symbols"]
        for kind, items in sorted(by_kind.items()):
            lines.append(f"\n{kind} ({len(items)}):")
            lines += [f"  {i.symbol}: {i.detail}" for i in items[:max_per_kind]]
            if len(items) > max_per_kind:
                lines.append(f"  … {len(items) - max_per_kind} more")
        return "\n".join(lines)


def _sample(values: list[str], n: int = 3) -> str:
    head = ", ".join(values[:n])
    return head + (f" (+{len(values) - n})" if len(values) > n else "")


def check_symbol(
    bars: pd.DataFrame,
    calendar: list[str],
    split_dates: set[str] | None = None,
    check_volume: bool = True,
) -> list[Issue]:
    """Checks one symbol's bars (sorted or not) against the trading calendar."""
    if bars.empty:
        return []
    symbol = str(bars["symbol"].iloc[0])
    split_dates = split_dates or set()
    issues: list[Issue] = []

    dup = bars[bars.duplicated(subset=["date"], keep=False)]["date"].unique().tolist()
    if dup:
        issues.append(Issue("duplicate", symbol, _sample(sorted(dup))))
    b = bars.drop_duplicates(subset=["date"], keep="last").sort_values("date")

    first, last = b["date"].iloc[0], b["date"].iloc[-1]
    have = set(b["date"])
    missing = [d for d in calendar if first <= d <= last and d not in have]
    if missing:
        issues.append(Issue("missing_days", symbol, f"{len(missing)} missing: {_sample(missing)}"))

    if calendar:
        later = [d for d in calendar if d > last]
        if len(later) >= STALE_SESSIONS:
            issues.append(Issue("stale", symbol, f"last bar {last}, {len(later)} sessions behind"))

    prices = b[["open", "high", "low", "close"]]
    bad = (
        prices.isna().any(axis=1)
        | (prices <= 0).any(axis=1)
        | (b["low"] > b[["open", "close"]].min(axis=1) + 1e-6)
        | (b["high"] < b[["open", "close"]].max(axis=1) - 1e-6)
    )
    if bad.any():
        issues.append(Issue("bad_ohlc", symbol, _sample(b.loc[bad, "date"].tolist())))

    if check_volume:
        zero = b.loc[b["volume"] <= 0, "date"].tolist()
        if zero:
            issues.append(Issue("zero_volume", symbol, f"{len(zero)} days: {_sample(zero)}"))

    ret = b["close"].pct_change().abs()
    jumps = b.loc[(ret > JUMP_THRESHOLD) & ~b["date"].isin(split_dates), "date"].tolist()
    if jumps:
        issues.append(
            Issue(
                "big_jump",
                symbol,
                f">{JUMP_THRESHOLD:.0%} move, check split/bonus: " + _sample(jumps),
            )
        )
    return issues


def check_frame(
    bars: pd.DataFrame,
    calendar: list[str],
    splits: pd.DataFrame | None = None,
    no_volume_symbols: set[str] | None = None,
) -> QualityReport:
    report = QualityReport()
    no_volume_symbols = no_volume_symbols or set()
    split_map: dict[str, set[str]] = defaultdict(set)
    if splits is not None and not splits.empty:
        for row in splits.itertuples(index=False):
            split_map[row.symbol].add(row.date)
    for symbol, group in bars.groupby("symbol"):
        report.checked += 1
        report.issues += check_symbol(
            group,
            calendar,
            split_map.get(str(symbol)),
            check_volume=symbol not in no_volume_symbols,
        )
    return report
