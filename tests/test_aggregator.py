from datetime import UTC, datetime

import polars as pl

from thestrat.aggregator import TimeframeAggregator


def _make_daily_bars(days: list[tuple]) -> pl.DataFrame:
    """Helper: create a Polars DataFrame of daily bars.
    Each tuple: (date_str, open, high, low, close, volume)
    """
    return pl.DataFrame(
        [
            {
                "timestamp": datetime.fromisoformat(d[0]).replace(tzinfo=UTC),
                "open": d[1],
                "high": d[2],
                "low": d[3],
                "close": d[4],
                "volume": d[5],
            }
            for d in days
        ]
    )


def test_aggregate_weekly():
    """Aggregates daily bars into weekly bars (Monday-aligned)."""
    bars = _make_daily_bars(
        [
            ("2026-04-06T20:00:00", 100.0, 105.0, 99.0, 104.0, 1000.0),
            ("2026-04-07T20:00:00", 104.0, 108.0, 103.0, 107.0, 1200.0),
            ("2026-04-08T20:00:00", 107.0, 110.0, 106.0, 109.0, 1100.0),
            ("2026-04-09T20:00:00", 109.0, 112.0, 108.0, 111.0, 1300.0),
            ("2026-04-10T20:00:00", 111.0, 115.0, 110.0, 114.0, 1400.0),
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1w")
    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["open"] == 100.0
    assert row["high"] == 115.0
    assert row["low"] == 99.0
    assert row["close"] == 114.0
    assert row["volume"] == 6000.0


def test_aggregate_monthly():
    bars = _make_daily_bars(
        [
            ("2026-04-01T20:00:00", 100.0, 105.0, 99.0, 104.0, 1000.0),
            ("2026-04-15T20:00:00", 104.0, 110.0, 103.0, 108.0, 1200.0),
            ("2026-05-01T20:00:00", 108.0, 112.0, 107.0, 111.0, 1100.0),
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1m")
    assert len(result) == 2
    april = result.filter(pl.col("timestamp").dt.month() == 4).row(0, named=True)
    assert april["open"] == 100.0
    assert april["high"] == 110.0
    assert april["low"] == 99.0
    assert april["close"] == 108.0
    assert april["volume"] == 2200.0


def test_aggregate_quarterly():
    bars = _make_daily_bars(
        [
            ("2026-01-15T20:00:00", 100.0, 105.0, 99.0, 104.0, 1000.0),
            ("2026-03-15T20:00:00", 104.0, 110.0, 103.0, 108.0, 1200.0),
            ("2026-04-15T20:00:00", 108.0, 112.0, 107.0, 111.0, 1100.0),
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1q")
    assert len(result) == 2


def test_aggregate_yearly():
    bars = _make_daily_bars(
        [
            ("2025-06-15T20:00:00", 100.0, 105.0, 99.0, 104.0, 1000.0),
            ("2026-03-15T20:00:00", 108.0, 112.0, 107.0, 111.0, 1100.0),
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1y")
    assert len(result) == 2


def _make_5sec_bars(entries: list[tuple]) -> pl.DataFrame:
    """Helper: create 5sec bars. Each tuple: (iso_timestamp, open, high, low, close, volume)"""
    return pl.DataFrame(
        [
            {
                "timestamp": datetime.fromisoformat(e[0]).replace(tzinfo=UTC),
                "open": e[1],
                "high": e[2],
                "low": e[3],
                "close": e[4],
                "volume": e[5],
            }
            for e in entries
        ]
    )


def test_aggregate_empty():
    bars = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1w")
    assert len(result) == 0


def test_aggregate_1min_from_5sec():
    """12 x 5sec bars -> 1 x 1min bar."""
    bars = _make_5sec_bars(
        [(f"2026-04-07T13:30:{s:02d}", 100.0 + s, 101.0 + s, 99.0, 100.5 + s, 1000.0) for s in range(0, 60, 5)]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1min")
    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["open"] == 100.0
    assert row["close"] == 155.5
    assert row["volume"] == 12000.0


def test_aggregate_5min_from_5sec():
    """5sec bars spanning 10 minutes -> 2 x 5min bars."""
    bars_list = []
    for minute in range(30, 40):
        for s in range(0, 60, 5):
            bars_list.append((f"2026-04-07T13:{minute}:{s:02d}", 100.0, 101.0, 99.0, 100.0, 100.0))
    bars = _make_5sec_bars(bars_list)
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "5min")
    assert len(result) == 2


def test_aggregate_1h_equity_offset():
    """1h bars with equity offset: boundaries at :30."""
    bars = _make_5sec_bars(
        [
            ("2026-04-07T13:30:00", 100.0, 101.0, 99.0, 100.0, 100.0),
            ("2026-04-07T13:59:55", 102.0, 103.0, 101.0, 102.0, 100.0),
            ("2026-04-07T14:00:00", 102.0, 104.0, 101.0, 103.0, 100.0),
            ("2026-04-07T14:29:55", 103.0, 105.0, 102.0, 104.0, 100.0),
            ("2026-04-07T14:30:00", 104.0, 106.0, 103.0, 105.0, 100.0),
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1h", equity_offset=True)
    assert len(result) == 2  # 13:30-14:29 and 14:30+


def test_aggregate_1h_no_offset():
    """1h bars without offset: boundaries at :00."""
    bars = _make_5sec_bars(
        [
            ("2026-04-07T13:00:00", 100.0, 101.0, 99.0, 100.0, 100.0),
            ("2026-04-07T13:59:55", 102.0, 103.0, 101.0, 102.0, 100.0),
            ("2026-04-07T14:00:00", 102.0, 104.0, 101.0, 103.0, 100.0),
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "1h", equity_offset=False)
    assert len(result) == 2


def test_aggregate_4h_equity_offset():
    bars = _make_5sec_bars(
        [
            ("2026-04-07T13:30:00", 100.0, 101.0, 99.0, 100.0, 100.0),
            ("2026-04-07T17:29:55", 110.0, 111.0, 109.0, 110.0, 100.0),
            ("2026-04-07T17:30:00", 110.0, 112.0, 109.0, 111.0, 100.0),
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "4h", equity_offset=True)
    assert len(result) == 2


def test_aggregate_6h():
    bars = _make_5sec_bars(
        [
            ("2026-04-07T00:00:00", 100.0, 101.0, 99.0, 100.0, 100.0),
            ("2026-04-07T05:59:55", 105.0, 106.0, 104.0, 105.0, 100.0),
            ("2026-04-07T06:00:00", 105.0, 107.0, 104.0, 106.0, 100.0),
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "6h", equity_offset=False)
    assert len(result) == 2


def test_aggregate_12h():
    bars = _make_5sec_bars(
        [
            ("2026-04-07T00:00:00", 100.0, 101.0, 99.0, 100.0, 100.0),
            ("2026-04-07T11:59:55", 105.0, 106.0, 104.0, 105.0, 100.0),
            ("2026-04-07T12:00:00", 105.0, 107.0, 104.0, 106.0, 100.0),
        ]
    )
    agg = TimeframeAggregator()
    result = agg.aggregate(bars, "12h", equity_offset=False)
    assert len(result) == 2


def test_existing_daily_aggregation_still_works():
    """Verify existing daily+ aggregation (1w, 1m, 1q, 1y) still works with new signature."""
    bars = _make_daily_bars(
        [
            ("2026-04-06T20:00:00", 100.0, 105.0, 99.0, 104.0, 1000.0),
            ("2026-04-07T20:00:00", 104.0, 108.0, 103.0, 107.0, 1200.0),
        ]
    )
    agg = TimeframeAggregator()
    # Without equity_offset (default)
    result = agg.aggregate(bars, "1w")
    assert len(result) == 1
    # With equity_offset explicitly False
    result = agg.aggregate(bars, "1w", equity_offset=False)
    assert len(result) == 1
