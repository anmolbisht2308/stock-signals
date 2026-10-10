"""Yahoo Finance via yfinance. Personal use only (Yahoo's terms).

Yahoo's OHLC is already split-adjusted; `Adj Close` also adjusts for dividends.
Bonus issues usually appear as splits, but not always — the quality checks flag
large unexplained jumps.
"""

import logging
import time
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd

from stock_signals.dates import IST
from stock_signals.sources import BAR_COLUMNS, SPLIT_COLUMNS, FetchResult

log = logging.getLogger(__name__)

INDEX_TICKERS = {"NIFTY50": "^NSEI", "NIFTY500": "^CRSLDX", "INDIAVIX": "^INDIAVIX"}

Downloader = Callable[..., pd.DataFrame]


def to_ticker(symbol: str) -> str:
    return INDEX_TICKERS.get(symbol, f"{symbol}.NS")


def _yf_download(**kwargs: object) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(**kwargs)


def _extract(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker not in raw.columns.get_level_values(0):
            return None
        return raw[ticker]
    return raw


def _date_strings(index: pd.Index) -> pd.Index:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(IST)
    return idx.strftime("%Y-%m-%d")


def normalise(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """yfinance frame for one ticker -> (bars, splits) in our column layout."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")
    dates = _date_strings(df.index)
    bars = pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": df["Open"].to_numpy(dtype=float),
            "high": df["High"].to_numpy(dtype=float),
            "low": df["Low"].to_numpy(dtype=float),
            "close": df["Close"].to_numpy(dtype=float),
            "adj_close": df.get("Adj Close", df["Close"]).to_numpy(dtype=float),
            "volume": df["Volume"].fillna(0).to_numpy(dtype=float),
        }
    )
    bars = bars.dropna(subset=["close"]).drop_duplicates(subset=["date"], keep="last")

    splits = pd.DataFrame(columns=SPLIT_COLUMNS)
    if "Stock Splits" in df.columns:
        s = df["Stock Splits"].fillna(0)
        mask = (s > 0).to_numpy()
        if mask.any():
            splits = pd.DataFrame(
                {"symbol": symbol, "date": dates[mask], "ratio": s[mask].to_numpy(dtype=float)}
            )
    return bars[BAR_COLUMNS].reset_index(drop=True), splits


class YahooSource:
    name = "yahoo"

    def __init__(
        self, download: Downloader = _yf_download, retries: int = 3, pause: float = 2.0
    ) -> None:
        self._download = download
        self.retries = retries
        self.pause = pause

    def _download_with_retry(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        for attempt in range(1, self.retries + 1):
            try:
                return self._download(
                    tickers=tickers,
                    start=start.isoformat(),
                    # yfinance's end is exclusive.
                    end=(end + timedelta(days=1)).isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
            except Exception as exc:  # network/parse errors from yfinance are untyped
                log.warning("yahoo download failed (attempt %d): %s", attempt, exc)
                if attempt == self.retries:
                    raise
                time.sleep(self.pause * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    def fetch(self, symbols: list[str], start: date, end: date) -> FetchResult:
        result = FetchResult()
        if not symbols:
            return result
        tickers = {to_ticker(s): s for s in symbols}
        raw = self._download_with_retry(list(tickers), start, end)
        bars, splits = [], []
        for ticker, symbol in tickers.items():
            df = _extract(raw, ticker)
            if df is None:
                result.failed.append(symbol)
                continue
            b, s = normalise(df, symbol)
            if b.empty:
                result.failed.append(symbol)
                continue
            bars.append(b)
            if not s.empty:
                splits.append(s)
        if bars:
            result.bars = pd.concat(bars, ignore_index=True)
        if splits:
            result.splits = pd.concat(splits, ignore_index=True)
        return result
