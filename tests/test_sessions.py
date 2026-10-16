"""Exchange eligibility and accepted local anchors through the public API."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from thestrat import SESSIONS, InstrumentType, ScheduleRangeError, Session, TheStrat, session_for


def utc(y, m, d, h=0, minute=0):
    return datetime(y, m, d, h, minute, tzinfo=UTC)


def bars(timestamps):
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [10.0] * len(timestamps),
            "high": [12.0] * len(timestamps),
            "low": [8.0] * len(timestamps),
            "close": [11.0] * len(timestamps),
            "volume": [1.0] * len(timestamps),
        }
    )


@pytest.mark.parametrize(
    "kind,tz,minutes",
    [("equity_us", "America/New_York", 570), ("futures_cme", "America/New_York", 1080), ("crypto", "UTC", 0)],
)
def test_presets(kind, tz, minutes):
    session = session_for(kind)
    assert session == SESSIONS[InstrumentType(kind)]
    assert (session.timezone, session.anchor_minutes) == (tz, minutes)


def test_fx_requires_product_calendar():
    with pytest.raises(ValueError, match="explicit Session"):
        session_for("fx")


@pytest.mark.parametrize(
    "anchor,expected",
    [(570, [utc(2026, 6, 8, 13, 30), utc(2026, 6, 8, 17, 30)]), (540, [utc(2026, 6, 8, 13), utc(2026, 6, 8, 17)])],
)
def test_four_hour_anchor_and_close_clipping(anchor, expected):
    regular = session_for("equity_us")
    s = TheStrat(
        session=Session(regular.timezone, anchor, regular.schedule),
        timeframes=["4h", "6h", "12h", "1d"],
        bars=bars(
            [
                utc(2026, 6, 8, 13, 29),
                utc(2026, 6, 8, 13, 35),
                utc(2026, 6, 8, 17, 35),
                utc(2026, 6, 8, 19, 59),
                utc(2026, 6, 8, 20),
            ]
        ),
    )
    result = s.aggregate()
    four = result.filter(pl.col("timeframe") == "4h")
    assert four["timestamp"].to_list() == expected
    assert four["volume"].to_list() == [1.0, 2.0]
    assert result.filter(pl.col("timeframe") == "12h")["volume"].to_list() == [3.0]
    assert result.filter(pl.col("timeframe") == "1d")["volume"].to_list() == [3.0]
    assert s.bars.filter(pl.col("timeframe") == "1min").height == 3


def test_early_close_holiday_weekend_and_forward_lookup():
    s = TheStrat(session=session_for("equity_us"), timeframes=["4h"])
    # Thanksgiving closed; Friday closes 13:00 ET, so no 13:30 4h start.
    s.bars = bars([utc(2026, 11, 26, 15), utc(2026, 11, 27, 14, 30), utc(2026, 11, 27, 17, 59), utc(2026, 11, 27, 18)])
    result = s.aggregate()
    assert result["timestamp"].to_list() == [utc(2026, 11, 27, 14, 30)]
    assert result["volume"].to_list() == [2.0]
    assert s.window(utc(2026, 11, 28, 12), after=1)["window_start"].to_list() == [utc(2026, 11, 30, 14, 30)]


def test_futures_midnight_and_break():
    s = TheStrat(
        session=session_for("futures_cme"),
        timeframes=["1d"],
        bars=bars(
            [utc(2026, 6, 7, 22), utc(2026, 6, 8, 3), utc(2026, 6, 8, 20, 59), utc(2026, 6, 8, 21), utc(2026, 6, 8, 22)]
        ),
    )
    result = s.aggregate()
    assert result["timestamp"].to_list() == [utc(2026, 6, 7, 22), utc(2026, 6, 8, 22)]
    assert result["volume"].to_list() == [3.0, 1.0]


@pytest.mark.parametrize("opened", [utc(2025, 3, 9, 22), utc(2025, 11, 2, 23)])
def test_futures_dst_open_wall_clock(opened):
    s = TheStrat(
        session=session_for("futures_cme"), timeframes=["1d"], bars=bars([opened, opened + timedelta(hours=5)])
    )
    assert s.aggregate()["timestamp"].to_list() == [opened]


def test_calendar_period_not_split_by_sessions():
    s = TheStrat(
        session=session_for("equity_us"),
        timeframes=["1w", "1m", "1q", "1y"],
        bars=bars([utc(2026, 6, 8, 13, 30), utc(2026, 6, 9, 13, 30)]),
    )
    result = s.aggregate()
    assert result["volume"].to_list() == [2.0] * 4
    assert result["timestamp"].to_list() == [
        utc(2026, 6, 8, 13, 30),
        utc(2026, 6, 1, 13, 30),
        utc(2026, 4, 1, 13, 30),
        utc(2026, 1, 1, 14, 30),
    ]


def test_crypto_weekend_and_naive_storage():
    s = TheStrat(
        session=session_for("crypto"),
        timeframes=["1d"],
        bars=bars([datetime(2026, 6, 7, 23, 59), datetime(2026, 6, 8)]),
    )
    result = s.aggregate()
    assert result["timestamp"].to_list() == [utc(2026, 6, 7), utc(2026, 6, 8)]
    assert result.schema["timestamp"] == pl.Datetime("us", "UTC")


def test_calendar_range_error():
    s = TheStrat(session=session_for("equity_us"))
    with pytest.raises(ScheduleRangeError):
        s.window(utc(2200, 1, 1))


def test_session_configuration_validation_and_frozen():
    with pytest.raises(ValueError):
        Session("UTC", True, session_for("crypto").schedule)
    with pytest.raises((AttributeError, Exception)):
        session_for("crypto").anchor_minutes = 60  # type: ignore[misc]
    regular = session_for("equity_us")
    s = TheStrat(session=Session(regular.timezone, 600, regular.schedule), bars=bars([utc(2026, 6, 8, 15)]))
    with pytest.raises(ValueError, match="anchor"):
        s.aggregate()


def test_session_value_equality_and_inequality():
    schedule = session_for("crypto").schedule
    assert Session("UTC", 0, schedule) == Session("UTC", 0, schedule)
    assert Session("UTC", 0, schedule) != Session("UTC", 60, schedule)
