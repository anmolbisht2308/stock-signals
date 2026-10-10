"""Market regime from the Nifty 50 trend and breadth.

risk_on   Nifty above its 200-day average and ≥ 40% of stocks above their 50-day average
caution   Nifty above its 200-day average but breadth below 40%
risk_off  Nifty below its 200-day average
"""

from dataclasses import dataclass

import pandas as pd

BREADTH_MIN = 0.40


@dataclass(frozen=True)
class Regime:
    date: str
    mode: str
    nifty_close: float
    nifty_sma200: float
    breadth50: float

    @property
    def swing_multiplier(self) -> float:
        return 1.0 if self.mode == "risk_on" else 0.5

    @property
    def positional_entries_allowed(self) -> bool:
        return self.mode != "risk_off"

    def describe(self) -> str:
        icon = {"risk_on": "🟢", "caution": "🟡", "risk_off": "🔴"}[self.mode]
        return (
            f"{icon} Market {self.mode.replace('_', '-')}: Nifty {self.nifty_close:,.0f} vs "
            f"200-DMA {self.nifty_sma200:,.0f}, breadth {self.breadth50:.0%} above 50-DMA"
        )


def regime_table(panel: pd.DataFrame, index_symbol: str, equities: set[str]) -> pd.DataFrame:
    """One row per date: nifty_close, nifty_sma200, breadth50, mode."""
    idx = panel[panel["symbol"] == index_symbol].set_index("date")
    eq = panel[panel["symbol"].isin(equities) & panel["sma50"].notna()]
    breadth = (eq["close"] > eq["sma50"]).groupby(eq["date"]).mean()
    out = pd.DataFrame(
        {"nifty_close": idx["close"], "nifty_sma200": idx["sma200"], "breadth50": breadth}
    ).dropna(subset=["nifty_close"])
    out["breadth50"] = out["breadth50"].fillna(0.0)

    def mode(row: pd.Series) -> str:
        if pd.isna(row.nifty_sma200) or row.nifty_close < row.nifty_sma200:
            return "risk_off"
        return "risk_on" if row.breadth50 >= BREADTH_MIN else "caution"

    out["mode"] = out.apply(mode, axis=1)
    return out


def regime_on(table: pd.DataFrame, date: str) -> Regime:
    row = table.loc[date]
    return Regime(
        date=date,
        mode=str(row["mode"]),
        nifty_close=float(row["nifty_close"]),
        nifty_sma200=float(row["nifty_sma200"]) if pd.notna(row["nifty_sma200"]) else 0.0,
        breadth50=float(row["breadth50"]),
    )
