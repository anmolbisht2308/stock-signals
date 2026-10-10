from datetime import date

import mongomock
import pandas as pd
import pytest

from stock_signals.sources import BAR_COLUMNS, SPLIT_COLUMNS, FetchResult


def weekdays(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def make_bars(
    symbol: str, days: list[str], start_price: float = 100.0, volume: float = 1000.0
) -> pd.DataFrame:
    rows = []
    price = start_price
    for i, d in enumerate(days):
        price = start_price + i * 0.5
        rows.append([symbol, d, price, price + 2, price - 2, price + 1, price + 1, volume])
    return pd.DataFrame(rows, columns=BAR_COLUMNS)


class FakeSource:
    name = "fake"

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        splits: pd.DataFrame | None = None,
        fail: set[str] | None = None,
    ) -> None:
        self.data = data
        self.splits = splits if splits is not None else pd.DataFrame(columns=SPLIT_COLUMNS)
        self.fail = fail or set()
        self.calls: list[tuple[list[str], date, date]] = []

    def fetch(self, symbols, start, end):
        self.calls.append((list(symbols), start, end))
        s, e = start.isoformat(), end.isoformat()
        res = FetchResult()
        frames = []
        for sym in symbols:
            df = self.data.get(sym)
            if sym in self.fail or df is None:
                res.failed.append(sym)
                continue
            frames.append(df[(df["date"] >= s) & (df["date"] <= e)])
        if frames:
            res.bars = pd.concat(frames, ignore_index=True)
        sp = self.splits
        res.splits = sp[sp["symbol"].isin(symbols) & (sp["date"] >= s) & (sp["date"] <= e)]
        return res


@pytest.fixture
def db():
    return mongomock.MongoClient(tz_aware=True)["test"]
