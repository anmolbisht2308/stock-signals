"""Dates are IST `YYYY-MM-DD` strings; timestamps are UTC."""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Daily bars are only treated as final after this time (market closes 15:30).
BAR_FINAL_AFTER = time(16, 0)


def now_ist() -> datetime:
    return datetime.now(IST)


def last_complete_session(now: datetime | None = None) -> date:
    """Latest date whose daily bar is final. Before 16:00 IST today's bar is still forming."""
    now = (now or now_ist()).astimezone(IST)
    if now.time() >= BAR_FINAL_AFTER:
        return now.date()
    return now.date() - timedelta(days=1)


def to_str(d: date) -> str:
    return d.isoformat()


def parse(s: str) -> date:
    return date.fromisoformat(s)
