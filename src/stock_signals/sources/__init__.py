from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import pandas as pd

BAR_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
SPLIT_COLUMNS = ["symbol", "date", "ratio"]


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLUMNS)


def empty_splits() -> pd.DataFrame:
    return pd.DataFrame(columns=SPLIT_COLUMNS)


@dataclass
class FetchResult:
    """Daily bars (split-adjusted OHLC, dividend-adjusted `adj_close`) plus split events."""

    bars: pd.DataFrame = field(default_factory=empty_bars)
    splits: pd.DataFrame = field(default_factory=empty_splits)
    failed: list[str] = field(default_factory=list)


class PriceSource(Protocol):
    name: str

    def fetch(self, symbols: list[str], start: date, end: date) -> FetchResult:
        """Bars for `start`..`end` inclusive."""
        ...
