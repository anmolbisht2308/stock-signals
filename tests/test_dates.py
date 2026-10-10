from datetime import date, datetime

from stock_signals.dates import IST, last_complete_session


def test_bar_is_final_only_after_16_ist():
    assert last_complete_session(datetime(2026, 10, 9, 15, 59, tzinfo=IST)) == date(2026, 10, 8)
    assert last_complete_session(datetime(2026, 10, 9, 16, 0, tzinfo=IST)) == date(2026, 10, 9)


def test_utc_input_is_converted_to_ist():
    from datetime import UTC

    # 11:00 UTC = 16:30 IST
    assert last_complete_session(datetime(2026, 10, 9, 11, 0, tzinfo=UTC)) == date(2026, 10, 9)
