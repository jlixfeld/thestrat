"""Aggregation accuracy tests with deterministic mock data and pre-computed expected values.

Tests validate OHLCV semantics (open=first, high=max, low=min, close=last, volume=sum)
across all timeframes using both monotonic and non-monotonic price patterns.
"""

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from thestrat.aggregator import TimeframeAggregator

# -- Helpers --

BASE_UTC = datetime(2026, 4, 7, 13, 30, 0, tzinfo=UTC)  # 9:30 ET = 13:30 UTC (EDT)


def _make_monotonic_intraday() -> pl.DataFrame:
    """Generate 4,680 monotonic 5sec bars for a full trading day (9:30-16:00 ET).

    Per-bar formula for bar j (0..11) in minute m (0..389):
      open  = m + j * 0.02
      high  = m + 0.5
      low   = m - 0.5
      close = m + 0.25 if j == 11 else m + j * 0.02 + 0.01
      volume = 100

    Note: minute m=0 produces low=-0.5 (negative prices are acceptable in synthetic test data).
    """
    rows = []
    for m in range(390):
        for j in range(12):
            ts = BASE_UTC + timedelta(minutes=m, seconds=j * 5)
            rows.append(
                {
                    "timestamp": ts,
                    "open": m + j * 0.02,
                    "high": m + 0.5,
                    "low": m - 0.5,
                    "close": m + 0.25 if j == 11 else m + j * 0.02 + 0.01,
                    "volume": 100.0,
                }
            )
    return pl.DataFrame(rows)


def _make_daily_5y() -> tuple[pl.DataFrame, dict[str, int]]:
    """Generate ~1,305 weekday-only daily bars from Jan 4 2021 to Jan 2 2026.

    Per-day formula (day sequence d = 0, 1, 2, ...):
      open   = d
      high   = d + 5
      low    = max(d - 5, 0)
      close  = d + 2
      volume = 1_000_000

    Returns (dataframe, lookup) where lookup maps "YYYY-MM-DD" -> d.
    """
    start = date(2021, 1, 4)  # Monday
    end = date(2026, 1, 2)  # Friday
    rows = []
    lookup = {}
    d = 0
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            ts = datetime(current.year, current.month, current.day, 20, 0, 0, tzinfo=UTC)
            rows.append(
                {
                    "timestamp": ts,
                    "open": float(d),
                    "high": float(d + 5),
                    "low": float(max(d - 5, 0)),
                    "close": float(d + 2),
                    "volume": 1_000_000.0,
                }
            )
            lookup[current.isoformat()] = d
            d += 1
        current += timedelta(days=1)
    return pl.DataFrame(rows), lookup


def _expected_window(s: int, e: int) -> dict:
    """Compute expected OHLCV for a monotonic intraday window spanning minutes s..e."""
    return {
        "open": float(s),
        "high": float(e + 0.5),
        "low": float(s - 0.5),
        "close": float(e + 0.25),
        "volume": float((e - s + 1) * 1200),
    }


def _expected_daily_window(s: int, e: int) -> dict:
    """Compute expected OHLCV for a daily window spanning day sequences s..e."""
    return {
        "open": float(s),
        "high": float(e + 5),
        "low": float(max(s - 5, 0)),
        "close": float(e + 2),
        "volume": float((e - s + 1) * 1_000_000),
    }


def _assert_ohlcv(row: dict, expected: dict, label: str = "") -> None:
    """Assert OHLCV values match expected, with helpful error messages."""
    prefix = f"{label}: " if label else ""
    assert row["open"] == pytest.approx(expected["open"]), f"{prefix}open mismatch"
    assert row["high"] == pytest.approx(expected["high"]), f"{prefix}high mismatch"
    assert row["low"] == pytest.approx(expected["low"]), f"{prefix}low mismatch"
    assert row["close"] == pytest.approx(expected["close"]), f"{prefix}close mismatch"
    assert row["volume"] == pytest.approx(expected["volume"]), f"{prefix}volume mismatch"


# -- Module-scoped fixtures --


@pytest.fixture(scope="module")
def monotonic_intraday_bars() -> pl.DataFrame:
    return _make_monotonic_intraday()


@pytest.fixture(scope="module")
def daily_5y() -> tuple[pl.DataFrame, dict[str, int]]:
    return _make_daily_5y()


# -- Non-Monotonic Correctness Tests --


def test_inverted_v_first_last_vs_min_max():
    """Non-monotonic data: verifies open=first (not min), close=last (not max),
    high=max (not last), low=min (not first)."""
    bars = pl.DataFrame(
        [
            {
                "timestamp": datetime(2026, 4, 7, 13, 30, 0, tzinfo=UTC),
                "open": 50.0,
                "high": 52.0,
                "low": 49.0,
                "close": 51.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 31, 0, tzinfo=UTC),
                "open": 55.0,
                "high": 58.0,
                "low": 54.0,
                "close": 57.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 32, 0, tzinfo=UTC),
                "open": 60.0,
                "high": 65.0,
                "low": 59.0,
                "close": 63.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 33, 0, tzinfo=UTC),
                "open": 45.0,
                "high": 48.0,
                "low": 42.0,
                "close": 46.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 34, 0, tzinfo=UTC),
                "open": 40.0,
                "high": 43.0,
                "low": 38.0,
                "close": 41.0,
                "volume": 1000.0,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "5min")
    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["open"] == 50.0  # first, NOT min (40)
    assert row["high"] == 65.0  # max, NOT last (43)
    assert row["low"] == 38.0  # min, NOT first (49)
    assert row["close"] == 41.0  # last, NOT max (63)
    assert row["volume"] == 5000.0


def test_non_monotonic_multi_window_isolation():
    """Two adjacent 5min windows: ascending then descending. Values don't leak across."""
    bars = pl.DataFrame(
        [
            # Window 1 (13:30-13:34): ascending
            {
                "timestamp": datetime(2026, 4, 7, 13, 30, 0, tzinfo=UTC),
                "open": 10.0,
                "high": 15.0,
                "low": 9.0,
                "close": 12.0,
                "volume": 100.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 31, 0, tzinfo=UTC),
                "open": 15.0,
                "high": 20.0,
                "low": 14.0,
                "close": 18.0,
                "volume": 100.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 32, 0, tzinfo=UTC),
                "open": 20.0,
                "high": 25.0,
                "low": 19.0,
                "close": 22.0,
                "volume": 100.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 33, 0, tzinfo=UTC),
                "open": 25.0,
                "high": 30.0,
                "low": 24.0,
                "close": 28.0,
                "volume": 100.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 34, 0, tzinfo=UTC),
                "open": 30.0,
                "high": 35.0,
                "low": 29.0,
                "close": 32.0,
                "volume": 100.0,
            },
            # Window 2 (13:35-13:39): descending (higher values than window 1)
            {
                "timestamp": datetime(2026, 4, 7, 13, 35, 0, tzinfo=UTC),
                "open": 90.0,
                "high": 95.0,
                "low": 88.0,
                "close": 92.0,
                "volume": 200.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 36, 0, tzinfo=UTC),
                "open": 80.0,
                "high": 85.0,
                "low": 78.0,
                "close": 82.0,
                "volume": 200.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 37, 0, tzinfo=UTC),
                "open": 70.0,
                "high": 75.0,
                "low": 68.0,
                "close": 72.0,
                "volume": 200.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 38, 0, tzinfo=UTC),
                "open": 60.0,
                "high": 65.0,
                "low": 58.0,
                "close": 62.0,
                "volume": 200.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 39, 0, tzinfo=UTC),
                "open": 50.0,
                "high": 55.0,
                "low": 48.0,
                "close": 52.0,
                "volume": 200.0,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "5min")
    assert len(result) == 2

    w1 = result.row(0, named=True)
    assert w1["open"] == 10.0
    assert w1["high"] == 35.0  # NOT 95.0 from window 2
    assert w1["low"] == 9.0
    assert w1["close"] == 32.0
    assert w1["volume"] == 500.0

    w2 = result.row(1, named=True)
    assert w2["open"] == 90.0
    assert w2["high"] == 95.0  # NOT 35.0 from window 1
    assert w2["low"] == 48.0
    assert w2["close"] == 52.0
    assert w2["volume"] == 1000.0


def test_non_monotonic_1h_equity_offset():
    """Non-monotonic prices across two 1h equity-offset windows."""
    bars = pl.DataFrame(
        [
            # Window 1 (group key 13:30 UTC = 9:30 ET)
            {
                "timestamp": datetime(2026, 4, 7, 13, 35, 0, tzinfo=UTC),
                "open": 50.0,
                "high": 55.0,
                "low": 48.0,
                "close": 52.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 50, 0, tzinfo=UTC),
                "open": 60.0,
                "high": 65.0,
                "low": 58.0,
                "close": 62.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 14, 20, 0, tzinfo=UTC),
                "open": 40.0,
                "high": 45.0,
                "low": 38.0,
                "close": 42.0,
                "volume": 1000.0,
            },
            # Window 2 (group key 14:30 UTC = 10:30 ET)
            {
                "timestamp": datetime(2026, 4, 7, 14, 35, 0, tzinfo=UTC),
                "open": 70.0,
                "high": 75.0,
                "low": 68.0,
                "close": 72.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 14, 50, 0, tzinfo=UTC),
                "open": 80.0,
                "high": 85.0,
                "low": 78.0,
                "close": 82.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 15, 20, 0, tzinfo=UTC),
                "open": 30.0,
                "high": 35.0,
                "low": 28.0,
                "close": 32.0,
                "volume": 1000.0,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1h", equity_offset=True)
    assert len(result) == 2

    w1 = result.row(0, named=True)
    assert w1["open"] == 50.0
    assert w1["high"] == 65.0
    assert w1["low"] == 38.0
    assert w1["close"] == 42.0
    assert w1["volume"] == 3000.0

    w2 = result.row(1, named=True)
    assert w2["open"] == 70.0
    assert w2["high"] == 85.0
    assert w2["low"] == 28.0
    assert w2["close"] == 32.0
    assert w2["volume"] == 3000.0


# -- Full Day Intraday Tests --


def test_unsupported_timeframe_raises():
    """Unsupported timeframe string raises ValueError."""
    bars = pl.DataFrame(
        [
            {
                "timestamp": datetime(2026, 4, 7, 13, 30, 0, tzinfo=UTC),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
                "volume": 500.0,
            }
        ]
    )
    agg = TimeframeAggregator()
    with pytest.raises(ValueError, match="Unsupported"):
        agg.aggregate(bars, "2h")


# Parametrized: (timeframe, equity_offset, expected_windows, first_s, first_e, last_s, last_e)
INTRADAY_CASES = [
    ("1min", False, 390, 0, 0, 389, 389),
    ("5min", False, 78, 0, 4, 385, 389),
    ("15min", False, 26, 0, 14, 375, 389),
    ("30min", False, 13, 0, 29, 360, 389),
    ("1h", True, 7, 0, 59, 360, 389),
    ("1h", False, 7, 0, 29, 330, 389),
    ("4h", True, 2, 0, 179, 180, 389),
    ("4h", False, 2, 0, 149, 150, 389),
    ("6h", True, 2, 0, 299, 300, 389),
    ("6h", False, 2, 0, 269, 270, 389),
    ("12h", True, 1, 0, 389, 0, 389),
    ("12h", False, 1, 0, 389, 0, 389),
]


@pytest.mark.parametrize(
    "timeframe,equity_offset,expected_windows,first_s,first_e,last_s,last_e",
    INTRADAY_CASES,
    ids=[f"{tf}{'_offset' if off else ''}" for tf, off, *_ in INTRADAY_CASES],
)
def test_full_day_intraday(
    monotonic_intraday_bars,
    timeframe,
    equity_offset,
    expected_windows,
    first_s,
    first_e,
    last_s,
    last_e,
):
    """Full trading day aggregated to each intraday timeframe: validate count + OHLCV."""
    agg = TimeframeAggregator()
    result = agg.aggregate(monotonic_intraday_bars, timeframe, equity_offset=equity_offset)

    assert len(result) == expected_windows, (
        f"{timeframe} (offset={equity_offset}): expected {expected_windows} windows, got {len(result)}"
    )

    rows = result.sort("timestamp").to_dicts()

    # First window
    first_expected = _expected_window(first_s, first_e)
    _assert_ohlcv(rows[0], first_expected, f"{timeframe} first window m={first_s}..{first_e}")

    # Last window
    last_expected = _expected_window(last_s, last_e)
    _assert_ohlcv(rows[-1], last_expected, f"{timeframe} last window m={last_s}..{last_e}")


# -- Boundary and Edge Case Tests --


def test_equity_offset_boundary_crossing():
    """Bars at 10:29:55 and 10:30:00 ET land in different 1h equity-offset windows."""
    bars = pl.DataFrame(
        [
            {
                "timestamp": datetime(2026, 4, 7, 14, 29, 55, tzinfo=UTC),  # 10:29:55 ET
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
                "volume": 500.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 14, 30, 0, tzinfo=UTC),  # 10:30:00 ET
                "open": 106.0,
                "high": 110.0,
                "low": 105.0,
                "close": 109.0,
                "volume": 600.0,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1h", equity_offset=True)
    assert len(result) == 2

    rows = result.sort("timestamp").to_dicts()
    # First bar (10:29:55) in 9:30 window
    assert rows[0]["open"] == 100.0
    assert rows[0]["volume"] == 500.0
    # Second bar (10:30:00) in 10:30 window
    assert rows[1]["open"] == 106.0
    assert rows[1]["volume"] == 600.0


def test_partial_last_window():
    """7 minutes of data aggregated to 5min: 1 full window (5 min) + 1 partial (2 min)."""
    bars_list = []
    for m in range(7):
        for j in range(12):
            ts = BASE_UTC + timedelta(minutes=m, seconds=j * 5)
            bars_list.append(
                {
                    "timestamp": ts,
                    "open": float(m + j * 0.02),
                    "high": float(m + 0.5),
                    "low": float(m - 0.5),
                    "close": float(m + 0.25 if j == 11 else m + j * 0.02 + 0.01),
                    "volume": 100.0,
                }
            )
    bars = pl.DataFrame(bars_list)
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "5min")
    assert len(result) == 2

    rows = result.sort("timestamp").to_dicts()
    # Full window: minutes 0-4
    _assert_ohlcv(rows[0], _expected_window(0, 4), "full window m=0..4")
    # Partial window: minutes 5-6
    _assert_ohlcv(rows[1], _expected_window(5, 6), "partial window m=5..6")


def test_single_bar_window():
    """One bar aggregated to 1h returns that bar unchanged."""
    bars = pl.DataFrame(
        [
            {
                "timestamp": datetime(2026, 4, 7, 14, 0, 0, tzinfo=UTC),
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 5000.0,
            }
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1h")
    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["open"] == 150.0
    assert row["high"] == 155.0
    assert row["low"] == 148.0
    assert row["close"] == 153.0
    assert row["volume"] == 5000.0


def test_sparse_data_gaps():
    """5min window with bars at minutes 0, 2, 4 only (gaps at 1, 3)."""
    bars = pl.DataFrame(
        [
            {
                "timestamp": datetime(2026, 4, 7, 13, 30, 0, tzinfo=UTC),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
                "volume": 500.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 32, 0, tzinfo=UTC),
                "open": 106.0,
                "high": 112.0,
                "low": 103.0,
                "close": 110.0,
                "volume": 600.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 34, 0, tzinfo=UTC),
                "open": 108.0,
                "high": 109.0,
                "low": 101.0,
                "close": 107.0,
                "volume": 400.0,
            },
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "5min")
    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["open"] == 100.0  # first bar's open
    assert row["high"] == 112.0  # max of all highs (minute 2)
    assert row["low"] == 99.0  # min of all lows (minute 0)
    assert row["close"] == 107.0  # last bar's close
    assert row["volume"] == 1500.0  # sum


def test_out_of_order_input():
    """Shuffled bars produce identical result to sorted bars."""
    sorted_bars = pl.DataFrame(
        [
            {
                "timestamp": datetime(2026, 4, 7, 13, 30, 0, tzinfo=UTC),
                "open": 50.0,
                "high": 52.0,
                "low": 49.0,
                "close": 51.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 31, 0, tzinfo=UTC),
                "open": 55.0,
                "high": 58.0,
                "low": 54.0,
                "close": 57.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 32, 0, tzinfo=UTC),
                "open": 60.0,
                "high": 65.0,
                "low": 59.0,
                "close": 63.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 33, 0, tzinfo=UTC),
                "open": 45.0,
                "high": 48.0,
                "low": 42.0,
                "close": 46.0,
                "volume": 1000.0,
            },
            {
                "timestamp": datetime(2026, 4, 7, 13, 34, 0, tzinfo=UTC),
                "open": 40.0,
                "high": 43.0,
                "low": 38.0,
                "close": 41.0,
                "volume": 1000.0,
            },
        ]
    )
    # Shuffle: reverse order
    shuffled_bars = sorted_bars.reverse()

    agg = TimeframeAggregator()
    sorted_result = agg.aggregate(sorted_bars, "5min")
    shuffled_result = agg.aggregate(shuffled_bars, "5min")

    assert sorted_result.to_dicts() == shuffled_result.to_dicts()


# -- Daily+ Tests --


def test_daily_all_timeframes_with_spotchecks(daily_5y):
    """5 years of daily bars aggregated to 1w, 1m, 1q, 1y with count + spot-check."""
    bars, lookup = daily_5y
    agg = TimeframeAggregator()

    # Weekly
    weekly = agg.aggregate(bars, "1w")
    assert 260 <= len(weekly) <= 265  # ~261 weeks in 5 years

    # Spot-check first full week: Jan 4-8 2021 (d=0..4)
    first_week = weekly.sort("timestamp").row(0, named=True)
    _assert_ohlcv(first_week, _expected_daily_window(0, 4), "1w first week")

    # Monthly
    monthly = agg.aggregate(bars, "1m")
    assert 59 <= len(monthly) <= 62  # ~61 months (Jan 2021 - Jan 2026)

    # Spot-check January 2023
    jan_start = lookup["2023-01-02"]  # First weekday of Jan 2023
    jan_end = lookup["2023-01-31"]  # Last weekday of Jan 2023
    jan_bars = monthly.filter((pl.col("timestamp").dt.year() == 2023) & (pl.col("timestamp").dt.month() == 1))
    assert len(jan_bars) == 1
    _assert_ohlcv(jan_bars.row(0, named=True), _expected_daily_window(jan_start, jan_end), "1m Jan 2023")

    # Quarterly
    quarterly = agg.aggregate(bars, "1q")
    assert 20 <= len(quarterly) <= 22  # ~21 quarters

    # Spot-check Q1 2023 (Jan-Mar)
    q1_start = lookup["2023-01-02"]
    q1_end = lookup["2023-03-31"]
    q1_bars = quarterly.filter((pl.col("timestamp").dt.year() == 2023) & (pl.col("timestamp").dt.quarter() == 1))
    assert len(q1_bars) == 1
    _assert_ohlcv(q1_bars.row(0, named=True), _expected_daily_window(q1_start, q1_end), "1q Q1 2023")

    # Yearly
    yearly = agg.aggregate(bars, "1y")
    assert len(yearly) == 6  # 2021-2025 full + 2026 partial

    # Spot-check year 2023
    y23_start = lookup["2023-01-02"]
    y23_end = lookup["2023-12-29"]
    y23_bars = yearly.filter(pl.col("timestamp").dt.year() == 2023)
    assert len(y23_bars) == 1
    _assert_ohlcv(y23_bars.row(0, named=True), _expected_daily_window(y23_start, y23_end), "1y 2023")


def test_per_year_ohlcv(daily_5y):
    """Validate OHLCV for each of the 5 full years (2021-2025)."""
    bars, lookup = daily_5y
    agg = TimeframeAggregator()
    yearly = agg.aggregate(bars, "1y").sort("timestamp")
    rows = yearly.to_dicts()

    for year in range(2021, 2026):
        # Find first and last weekday of the year in our dataset
        first_date = None
        last_date = None
        for date_str, _ in lookup.items():
            if date_str.startswith(str(year)):
                if first_date is None or date_str < first_date:
                    first_date = date_str
                if last_date is None or date_str > last_date:
                    last_date = date_str

        s = lookup[first_date]
        e = lookup[last_date]
        year_row = next(r for r in rows if r["timestamp"].year == year)
        _assert_ohlcv(year_row, _expected_daily_window(s, e), f"year {year}")


def test_cross_year_week(daily_5y):
    """Week of Dec 29 2025 spans into Jan 2026 — should be one weekly bar."""
    bars, lookup = daily_5y
    agg = TimeframeAggregator()
    weekly = agg.aggregate(bars, "1w").sort("timestamp")

    # Dataset uses weekday-only filtering (no holiday exclusion), so Jan 1 is included
    # The last week should contain Dec 29, 30, 31 2025 and Jan 1, 2 2026
    # truncate("1w") groups all to Monday Dec 29
    last_week = weekly.row(-1, named=True)

    # Dec 29 2025 is present (Monday)
    d_start = lookup["2025-12-29"]
    # Jan 2 2026 is the last day in our dataset
    d_end = lookup["2026-01-02"]
    _assert_ohlcv(last_week, _expected_daily_window(d_start, d_end), "cross-year week")

    # Verify yearly aggregation still separates 2025 from 2026
    yearly = agg.aggregate(bars, "1y").sort("timestamp")
    year_2026 = yearly.filter(pl.col("timestamp").dt.year() == 2026)
    assert len(year_2026) == 1
    # 2026 should only contain Jan 1-2
    d_2026_start = lookup["2026-01-01"]
    d_2026_end = lookup["2026-01-02"]
    _assert_ohlcv(
        year_2026.row(0, named=True),
        _expected_daily_window(d_2026_start, d_2026_end),
        "year 2026",
    )
