"""Tests for the public `bucket_key_expr` free function.

`bucket_key_expr(timeframe, session)` is the supported, public form of the
grouping expression that `TimeframeAggregator` uses internally to assign each
raw bar to its higher-timeframe "bucket" (the value thestrat groups on and
calls `_group`). Downstream consumers import it instead of re-implementing the
session-anchored grouping logic.
"""

from datetime import UTC, datetime

import polars as pl
import pytest

from thestrat import SESSIONS, InstrumentType, bucket_key_expr
from thestrat.aggregator import TimeframeAggregator


def _utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _rth_minute_frame():
    """One RTH equity session (2026-06-08) as 1-minute UTC bars.

    09:30 ET = 13:30 UTC (EDT); 16:00 ET = 20:00 UTC. RTH is [13:30, 20:00),
    i.e. 390 one-minute bars.
    """
    start = _utc(2026, 6, 8, 13, 30)
    ts_series = pl.datetime_range(
        start=start,
        end=_utc(2026, 6, 8, 19, 59),
        interval="1m",
        time_zone="UTC",
        eager=True,
    )
    return pl.DataFrame({"timestamp": ts_series}).with_columns(
        pl.lit(1.0).alias("open"),
        pl.lit(1.0).alias("high"),
        pl.lit(1.0).alias("low"),
        pl.lit(1.0).alias("close"),
        pl.lit(1).alias("volume"),
    )


def _apply(expr: pl.Expr, frame: pl.DataFrame) -> pl.Series:
    return frame.with_columns(expr.alias("_group"))["_group"]


# ---------------------------------------------------------------------------
# (d) importable from the package top level
# ---------------------------------------------------------------------------


def test_bucket_key_expr_is_public():
    import thestrat

    assert hasattr(thestrat, "bucket_key_expr")
    assert bucket_key_expr in (thestrat.bucket_key_expr,)


# ---------------------------------------------------------------------------
# (a) EQUITY_US hourly anchors to the bottom of the hour (09:30/10:30/.../15:30)
# ---------------------------------------------------------------------------


def test_hourly_equity_us_anchors_to_bottom_of_hour():
    frame = _rth_minute_frame()
    groups = _apply(bucket_key_expr("1h", SESSIONS[InstrumentType.EQUITY_US]), frame)

    # 6.5h RTH → seven hourly buckets (the last is the 15:30–16:00 stub).
    assert groups.n_unique() == 7

    # Every bucket boundary lands on :30 ET, at hours 09..15.
    et = groups.unique().sort().dt.convert_time_zone("America/New_York")
    assert et.dt.minute().to_list() == [30] * 7
    assert et.dt.hour().to_list() == [9, 10, 11, 12, 13, 14, 15]


def test_daily_equity_us_anchors_to_930_et():
    frame = _rth_minute_frame()
    groups = _apply(bucket_key_expr("1d", SESSIONS[InstrumentType.EQUITY_US]), frame)

    # A single RTH session collapses to one daily bucket anchored at 09:30 ET.
    assert groups.n_unique() == 1
    et = groups.unique().dt.convert_time_zone("America/New_York")
    assert et.dt.hour().to_list() == [9]
    assert et.dt.minute().to_list() == [30]


# ---------------------------------------------------------------------------
# (b) FUTURES_CME session output matches the pre-refactor private method
#     row-for-row (parity guard).
# ---------------------------------------------------------------------------


def test_futures_cme_matches_private_group_expression():
    frame = _rth_minute_frame()
    session = SESSIONS[InstrumentType.FUTURES_CME]
    agg = TimeframeAggregator()

    for tf in ("1h", "4h", "1d", "1w"):
        legacy = _apply(agg._group_expression(tf, False, session), frame)
        public = _apply(bucket_key_expr(tf, session), frame)
        assert public.equals(legacy), f"{tf} diverged from _group_expression"


def test_no_session_matches_utc_truncation():
    frame = _rth_minute_frame()
    for tf in ("1h", "4h", "1d"):
        no_session = _apply(bucket_key_expr(tf, None), frame)
        truncated = _apply(pl.col("timestamp").dt.truncate({"1h": "1h", "4h": "4h", "1d": "1d"}[tf]), frame)
        assert no_session.equals(truncated), f"{tf} no-session path is not plain UTC truncation"


def test_sub_hour_ignores_session():
    """Sub-hour timeframes are never session-aligned — the session arg is inert."""
    frame = _rth_minute_frame()
    for tf, unit in (("1min", "1m"), ("5min", "5m"), ("15min", "15m"), ("30min", "30m")):
        with_session = _apply(bucket_key_expr(tf, SESSIONS[InstrumentType.EQUITY_US]), frame)
        truncated = _apply(pl.col("timestamp").dt.truncate(unit), frame)
        assert with_session.equals(truncated), f"{tf} sub-hour bucket is not plain truncation"


# ---------------------------------------------------------------------------
# (c) unknown timeframe raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("session", [None, SESSIONS[InstrumentType.EQUITY_US]])
def test_unknown_timeframe_raises(session):
    with pytest.raises(ValueError, match="timeframe"):
        bucket_key_expr("bogus", session)
