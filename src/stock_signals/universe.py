"""Stock universe (Nifty 200 / Nifty 500) and the `symbols` collection.

Constituent CSVs come from niftyindices.com (columns: Company Name, Industry,
Symbol, Series, ISIN Code). Refresh them manually every six months.
"""

import csv
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pymongo import ASCENDING
from pymongo.database import Database

log = logging.getLogger(__name__)

INDEX_URLS = {
    "NIFTY200": "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "NIFTY500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
}

# Market-wide series stored alongside stocks (not tradable universe members).
MARKET_SYMBOLS = {"NIFTY50": "Nifty 50 index", "INDIAVIX": "India VIX"}
CALENDAR_SYMBOL = "NIFTY50"


@dataclass(frozen=True)
class Member:
    symbol: str
    name: str
    industry: str
    isin: str


def universe_path(data_dir: Path, index: str) -> Path:
    return data_dir / "universe" / f"{index.lower()}.csv"


def download_universe(data_dir: Path, index: str) -> Path:
    path = universe_path(data_dir, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(
        INDEX_URLS[index],
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
        follow_redirects=True,
    )
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def read_universe(path: Path) -> list[Member]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    members = []
    for r in rows:
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if r.get("Series", "EQ") != "EQ" or not r.get("Symbol"):
            continue
        members.append(
            Member(
                symbol=r["Symbol"],
                name=r.get("Company Name", ""),
                industry=r.get("Industry", ""),
                isin=r.get("ISIN Code", ""),
            )
        )
    return members


class SymbolStore:
    """`symbols`: symbol, name, industry, isin, kind (equity|market), indices, status, liquidity.

    status: active | removed (left every tracked index) | renamed (see `renamed_to`).
    """

    def __init__(self, db: Database) -> None:
        self.col = db["symbols"]

    def ensure_indexes(self) -> None:
        self.col.create_index([("symbol", ASCENDING)], unique=True)

    def ensure_market_symbols(self) -> None:
        for symbol, name in MARKET_SYMBOLS.items():
            self.col.update_one(
                {"symbol": symbol},
                {"$set": {"name": name, "kind": "market", "status": "active"}},
                upsert=True,
            )

    def sync_index(self, index: str, members: list[Member]) -> dict[str, int]:
        """Make `index` membership match `members`. Returns added/removed counts."""
        now = datetime.now(UTC)
        wanted = {m.symbol for m in members}
        before = {d["symbol"] for d in self.col.find({"indices": index}, {"symbol": 1})}
        for m in members:
            self.col.update_one(
                {"symbol": m.symbol},
                {
                    "$set": {
                        "name": m.name,
                        "industry": m.industry,
                        "isin": m.isin,
                        "kind": "equity",
                        "status": "active",
                        "updated_at": now,
                    },
                    "$addToSet": {"indices": index},
                },
                upsert=True,
            )
        dropped = before - wanted
        if dropped:
            self.col.update_many({"symbol": {"$in": list(dropped)}}, {"$pull": {"indices": index}})
            self.col.update_many(
                {"symbol": {"$in": list(dropped)}, "indices": {"$size": 0}},
                {"$set": {"status": "removed", "updated_at": now}},
            )
        return {"members": len(wanted), "added": len(wanted - before), "removed": len(dropped)}

    def rename(self, old: str, new: str) -> None:
        """Record an NSE symbol change. The new symbol gets a full backfill on the next update."""
        doc = self.col.find_one({"symbol": old}, {"_id": 0}) or {}
        carry = {k: doc[k] for k in ("name", "industry", "isin", "kind", "indices") if k in doc}
        self.col.update_one(
            {"symbol": new},
            {"$set": {**carry, "status": "active", "renamed_from": old}},
            upsert=True,
        )
        self.col.update_one(
            {"symbol": old}, {"$set": {"status": "renamed", "renamed_to": new, "indices": []}}
        )

    def active(self, kind: str | None = None, index: str | None = None) -> list[str]:
        query: dict = {"status": "active"}
        if kind:
            query["kind"] = kind
        if index:
            query["indices"] = index
        return sorted(d["symbol"] for d in self.col.find(query, {"symbol": 1}))

    def set_fields(self, symbol: str, fields: dict) -> None:
        self.col.update_one({"symbol": symbol}, {"$set": fields})

    def get(self, symbol: str) -> dict | None:
        return self.col.find_one({"symbol": symbol}, {"_id": 0})
