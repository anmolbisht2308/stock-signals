"""`bars` collection: one document per (symbol, date).

A regular collection with a unique index (not a time-series collection) so that
re-running any download is an idempotent upsert.
"""

from datetime import UTC, datetime

import pandas as pd
from pymongo import ASCENDING, UpdateOne
from pymongo.database import Database

from stock_signals.sources import BAR_COLUMNS

PRICE_FIELDS = ["open", "high", "low", "close", "adj_close"]


class BarStore:
    def __init__(self, db: Database) -> None:
        self.col = db["bars"]

    def ensure_indexes(self) -> None:
        self.col.create_index([("symbol", ASCENDING), ("date", ASCENDING)], unique=True)
        self.col.create_index([("date", ASCENDING)])

    def upsert(self, bars: pd.DataFrame, source: str, batch: int = 5000) -> int:
        if bars.empty:
            return 0
        now = datetime.now(UTC)
        ops = []
        for row in bars[BAR_COLUMNS].itertuples(index=False):
            doc = {f: round(float(getattr(row, f)), 4) for f in PRICE_FIELDS}
            doc["volume"] = int(row.volume)
            doc["source"] = source
            doc["updated_at"] = now
            ops.append(
                UpdateOne({"symbol": row.symbol, "date": row.date}, {"$set": doc}, upsert=True)
            )
        for i in range(0, len(ops), batch):
            self.col.bulk_write(ops[i : i + batch], ordered=False)
        return len(ops)

    def delete_symbol(self, symbol: str) -> int:
        return self.col.delete_many({"symbol": symbol}).deleted_count

    def last_dates(self) -> dict[str, str]:
        rows = self.col.aggregate([{"$group": {"_id": "$symbol", "last": {"$max": "$date"}}}])
        return {r["_id"]: r["last"] for r in rows}

    def load(
        self,
        symbols: str | list[str],
        start: str | None = None,
        end: str | None = None,
        last_n: int | None = None,
    ) -> pd.DataFrame:
        """Bars sorted by symbol, date. `last_n` keeps only the latest n bars per symbol."""
        if isinstance(symbols, str):
            symbols = [symbols]
        query: dict = {"symbol": {"$in": symbols}}
        if start or end:
            query["date"] = {}
            if start:
                query["date"]["$gte"] = start
            if end:
                query["date"]["$lte"] = end
        docs = list(self.col.find(query, {"_id": 0, "source": 0, "updated_at": 0}))
        if not docs:
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.DataFrame(docs)[BAR_COLUMNS].sort_values(["symbol", "date"])
        if last_n:
            df = df.groupby("symbol", group_keys=False).tail(last_n)
        return df.reset_index(drop=True)

    def dates(self, symbol: str, start: str | None = None, end: str | None = None) -> list[str]:
        query: dict = {"symbol": symbol}
        if start or end:
            query["date"] = {k: v for k, v in (("$gte", start), ("$lte", end)) if v}
        return sorted(d["date"] for d in self.col.find(query, {"_id": 0, "date": 1}))
