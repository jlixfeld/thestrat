"""Tests for InstrumentType / Session presets and session-aware aggregation."""

from datetime import UTC, datetime

import polars as pl
import pytest

from thestrat import (
    SESSIONS,
    InstrumentType,
    Session,
    TimeframeAggregator,
    session_for,
)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_all_instrument_types_have_presets():
    assert set(SESSIONS.keys()) == set(InstrumentType)


@pytest.mark.parametrize(
    "kind,tz,minutes",
    [
        (InstrumentType.EQUITY_US, "America/New_York", 570),  # 9:30 ET
        (InstrumentType.FUTURES_CME, "America/New_York", 1080),  # 18:00 ET
        (InstrumentType.CRYPTO, "UTC", 0),
        (InstrumentType.FX, "America/New_York", 1020),  # 17:00 ET
    ],
)
def test_session_for_returns_canonical_preset(kind, tz, minutes):
    s = session_for(kind)
    assert s.timezone == tz
    assert s.anchor_minutes == minutes


# ---------------------------------------------------------------------------
# Session-aware daily aggregation
# ---------------------------------------------------------------------------


def _utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_daily_with_futures_cme_session_aligns_to_18_et():
    # CME futures session start is 18:00 ET = 22:00 UTC (during DST that's 23:00 UTC).
    # In June 2026 NY is on EDT, so 18:00 ET = 22:00 UTC.
    # A bar at 22:00 UTC on Sun should belong to Mon's session.
    bars = pl.DataFrame(
        [
            # Sun June 7 22:00 UTC = Sun June 7 18:00 EDT — session starts (Mon's session)
            {
                "timestamp": _utc(2026, 6, 7, 22, 0),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 100,
            },
            # Mon June 8 03:00 UTC = Sun 23:00 EDT — same session as above
            {
                "timestamp": _utc(2026, 6, 8, 3, 0),
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 200,
            },
            # Mon June 8 21:00 UTC = Mon 17:00 EDT — last bar of Mon's session
            {
                "timestamp": _utc(2026, 6, 8, 21, 0),
                "open": 101.5,
                "high": 103.0,
                "low": 101.0,
                "close": 102.5,
                "volume": 300,
            },
            # Mon June 8 22:00 UTC = Mon 18:00 EDT — session starts (Tue's session)
            {
                "timestamp": _utc(2026, 6, 8, 22, 0),
                "open": 102.5,
                "high": 104.0,
                "low": 102.0,
                "close": 103.5,
                "volume": 150,
            },
            # Tue June 9 02:00 UTC = Mon 22:00 EDT — same session as above (Tue)
            {
                "timestamp": _utc(2026, 6, 9, 2, 0),
                "open": 103.5,
                "high": 105.0,
                "low": 103.0,
                "close": 104.0,
                "volume": 250,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1d", session=SESSIONS[InstrumentType.FUTURES_CME]).sort("timestamp")

    # Expect 2 daily bars (Mon's session and Tue's session), not 3 from UTC midnight split
    assert len(result) == 2

    mon = result.row(0, named=True)
    tue = result.row(1, named=True)
    # Mon's session: open of first bar 100.0, close of last bar 102.5,
    # high = max of first 3 bars, low = min of first 3 bars, vol = sum
    assert mon["open"] == 100.0
    assert mon["high"] == 103.0
    assert mon["low"] == 99.0
    assert mon["close"] == 102.5
    assert mon["volume"] == 600
    # Tue's session: open 102.5 (the 22:00 UTC bar), close 104.0, vol = 400
    assert tue["open"] == 102.5
    assert tue["close"] == 104.0
    assert tue["volume"] == 400


def test_daily_with_equity_us_session_aligns_to_9_30_et():
    # Equity US daily anchor is 9:30 ET. June 8 9:30 EDT = June 8 13:30 UTC.
    # All bars in 13:30 UTC → next day 13:29 UTC are one daily bar.
    bars = pl.DataFrame(
        [
            # Mon 13:30 UTC = Mon 9:30 EDT — session starts (Mon's day)
            {
                "timestamp": _utc(2026, 6, 8, 13, 30),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 100,
            },
            # Mon 20:00 UTC = Mon 16:00 EDT — close of Mon's RTH
            {
                "timestamp": _utc(2026, 6, 8, 20, 0),
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 200,
            },
            # Tue 13:30 UTC = Tue 9:30 EDT — Tue's day starts
            {
                "timestamp": _utc(2026, 6, 9, 13, 30),
                "open": 101.5,
                "high": 103.0,
                "low": 101.0,
                "close": 102.5,
                "volume": 150,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1d", session=SESSIONS[InstrumentType.EQUITY_US]).sort("timestamp")

    assert len(result) == 2


def test_hourly_with_equity_us_session_aligns_to_30():
    # 60-min "Bottom of the Hour" buckets at 9:30, 10:30, 11:30 ET.
    # Mon 13:30 UTC = 9:30 EDT
    bars = pl.DataFrame(
        [
            {
                "timestamp": _utc(2026, 6, 8, 13, 30),
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 100,
            },
            {
                "timestamp": _utc(2026, 6, 8, 14, 0),
                "open": 100.5,
                "high": 101.5,
                "low": 100.0,
                "close": 101.0,
                "volume": 100,
            },
            {
                "timestamp": _utc(2026, 6, 8, 14, 29),
                "open": 101.0,
                "high": 101.6,
                "low": 100.8,
                "close": 101.4,
                "volume": 100,
            },
            # 14:30 UTC = 10:30 EDT — new bucket
            {
                "timestamp": _utc(2026, 6, 8, 14, 30),
                "open": 101.4,
                "high": 102.0,
                "low": 101.3,
                "close": 101.8,
                "volume": 100,
            },
            {
                "timestamp": _utc(2026, 6, 8, 15, 0),
                "open": 101.8,
                "high": 102.5,
                "low": 101.5,
                "close": 102.0,
                "volume": 100,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1h", session=SESSIONS[InstrumentType.EQUITY_US]).sort("timestamp")

    assert len(result) == 2
    first, second = result.row(0, named=True), result.row(1, named=True)
    # First bucket: 9:30-10:30 — contains the first 3 bars
    assert first["open"] == 100.0
    assert first["close"] == 101.4
    assert first["high"] == 101.6
    # Second bucket: 10:30-11:30 — contains the last 2 bars
    assert second["open"] == 101.4
    assert second["close"] == 102.0


def test_session_handles_dst_transition():
    # Spring-forward in NY is 2nd Sunday of March. In 2026 that's Mar 8.
    # Before Mar 8 NY is EST (UTC-5), after is EDT (UTC-4).
    # 18:00 EST Mar 6 = 23:00 UTC; 18:00 EDT Mar 9 = 22:00 UTC.
    # The session anchor MUST follow local 18:00, not a fixed UTC offset.
    bars = pl.DataFrame(
        [
            # Fri Mar 6 18:00 EST = 23:00 UTC. Start of Mar 9 session (Mon).
            {
                "timestamp": _utc(2026, 3, 6, 23, 0),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.2,
                "volume": 100,
            },
            # Mon Mar 9 18:00 EDT = 22:00 UTC. Start of Tue session.
            {
                "timestamp": _utc(2026, 3, 9, 22, 0),
                "open": 101.0,
                "high": 101.5,
                "low": 100.5,
                "close": 101.2,
                "volume": 100,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1d", session=SESSIONS[InstrumentType.FUTURES_CME])

    # Two distinct sessions, anchored at LOCAL 18:00 ET — DST handled.
    assert len(result) == 2


def test_aggregate_for_instrument_convenience():
    bars = pl.DataFrame(
        [
            {
                "timestamp": _utc(2026, 6, 7, 22, 0),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 100,
            },
            {
                "timestamp": _utc(2026, 6, 8, 21, 0),
                "open": 100.5,
                "high": 103.0,
                "low": 100.0,
                "close": 102.5,
                "volume": 300,
            },
            {
                "timestamp": _utc(2026, 6, 8, 22, 0),
                "open": 102.5,
                "high": 104.0,
                "low": 102.0,
                "close": 103.5,
                "volume": 150,
            },
        ]
    )
    agg = TimeframeAggregator()

    explicit = agg.aggregate(bars, "1d", session=SESSIONS[InstrumentType.FUTURES_CME]).sort("timestamp")
    convenience = agg.aggregate_for_instrument(bars, "1d", InstrumentType.FUTURES_CME).sort("timestamp")

    assert explicit.equals(convenience)


# ---------------------------------------------------------------------------
# Naive timestamps (the StratBacktester DuckDB case)
# ---------------------------------------------------------------------------


def test_session_aware_works_on_naive_timestamps():
    """DuckDB returns naive timestamps that are logically UTC. Aggregator
    should handle that without forcing the caller to add a timezone."""
    bars = pl.DataFrame(
        [
            {
                "timestamp": datetime(2026, 6, 7, 22, 0),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 100,
            },
            {
                "timestamp": datetime(2026, 6, 8, 21, 0),
                "open": 100.5,
                "high": 103.0,
                "low": 100.0,
                "close": 102.5,
                "volume": 300,
            },
            {
                "timestamp": datetime(2026, 6, 8, 22, 0),
                "open": 102.5,
                "high": 104.0,
                "low": 102.0,
                "close": 103.5,
                "volume": 150,
            },
        ]
    )
    ts_dtype = bars.schema["timestamp"]
    assert isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None  # naive

    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1d", session=SESSIONS[InstrumentType.FUTURES_CME]).sort("timestamp")

    # Two sessions
    assert len(result) == 2
    # Output timestamp stays naive (matches input shape)
    out_dtype = result.schema["timestamp"]
    assert isinstance(out_dtype, pl.Datetime) and out_dtype.time_zone is None


# ---------------------------------------------------------------------------
# Session is a frozen dataclass — equality + immutability
# ---------------------------------------------------------------------------


def test_session_is_frozen():
    s = Session(timezone="UTC", anchor_minutes=0)
    with pytest.raises((AttributeError, Exception)):
        s.anchor_minutes = 100  # type: ignore[misc]


def test_session_equality():
    a = Session(timezone="UTC", anchor_minutes=0)
    b = Session(timezone="UTC", anchor_minutes=0)
    c = Session(timezone="UTC", anchor_minutes=60)
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# Backward compat: legacy equity_offset still works
# ---------------------------------------------------------------------------


def test_legacy_equity_offset_still_aligns_hourly_to_30():
    bars = pl.DataFrame(
        [
            {
                "timestamp": _utc(2026, 6, 8, 13, 30),
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 100,
            },
            {
                "timestamp": _utc(2026, 6, 8, 14, 30),
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 100,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1h", equity_offset=True).sort("timestamp")
    # Each bar lands in its own 9:30/10:30 bucket
    assert len(result) == 2


def test_unsupported_timeframe_raises():
    bars = pl.DataFrame(
        [
            {
                "timestamp": _utc(2026, 6, 8, 13, 30),
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 100,
            },
        ]
    )
    agg = TimeframeAggregator()
    with pytest.raises(ValueError, match="Unsupported aggregation timeframe"):
        agg.aggregate(bars, "7min")
