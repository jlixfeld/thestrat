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
# DST transitions (wall-clock anchoring — regression for the physical-duration
# misanchor: the Globex Sunday session must not split on a spring-forward date)
# ---------------------------------------------------------------------------


def _minute_frame(start_utc: datetime, end_utc: datetime, interval: str = "1m") -> pl.DataFrame:
    ts = pl.datetime_range(start=start_utc, end=end_utc, interval=interval, time_zone="UTC", eager=True)
    return pl.DataFrame({"timestamp": ts})


def test_futures_spring_forward_sunday_session_is_one_daily_bucket():
    """US spring-forward 2025-03-09: the CME session (Sun 18:00 ET -> Mon 17:00 ET)
    spans the 02:00 transition. Physical-duration arithmetic split its first hour
    into a Saturday-keyed bucket; wall-clock arithmetic must keep one bucket
    keyed at the Sunday 18:00 ET open (22:00 UTC, EDT after the transition)."""
    frame = _minute_frame(_utc(2025, 3, 9, 22, 0), _utc(2025, 3, 10, 20, 59))
    groups = _apply(bucket_key_expr("1d", SESSIONS[InstrumentType.FUTURES_CME]), frame)

    assert groups.n_unique() == 1
    assert groups.unique().to_list() == [_utc(2025, 3, 9, 22, 0)]


def test_futures_fall_back_sunday_session_keyed_at_open():
    """US fall-back 2025-11-02: session opens Sun 18:00 EST = 23:00 UTC. The
    physical-duration arithmetic kept one bucket but keyed it 17:00 ET; the
    wall-clock key must be the 18:00 ET open."""
    frame = _minute_frame(_utc(2025, 11, 2, 23, 0), _utc(2025, 11, 3, 21, 59))
    groups = _apply(bucket_key_expr("1d", SESSIONS[InstrumentType.FUTURES_CME]), frame)

    assert groups.n_unique() == 1
    assert groups.unique().to_list() == [_utc(2025, 11, 2, 23, 0)]


def test_equity_rth_unaffected_by_dst_transitions():
    """RTH timestamps never cross the 02:00 Sunday transition when shifted by
    the 570-min anchor: hourly buckets on the Mondays after both 2025
    transitions still anchor 09:30..15:30 local."""
    for monday, utc_offset in (((2025, 3, 10), 4), ((2025, 11, 3), 5)):
        y, m, d = monday
        frame = _minute_frame(_utc(y, m, d, 9 + utc_offset, 30), _utc(y, m, d, 15 + utc_offset, 59))
        groups = _apply(bucket_key_expr("1h", SESSIONS[InstrumentType.EQUITY_US]), frame)
        et = groups.unique().sort().dt.convert_time_zone("America/New_York")
        assert et.dt.minute().to_list() == [30] * 7, f"{monday}: hourly buckets off :30"
        assert et.dt.hour().to_list() == [9, 10, 11, 12, 13, 14, 15], f"{monday}: wrong anchors"


def test_spring_forward_gap_bucket_key_shifts_past_gap():
    """A 4h futures bucket key of 02:00 local on a spring-forward date does not
    exist; it must shift forward to 03:00 EDT (07:00 UTC), not go null or raise."""
    # Sun 2025-03-09 03:30 / 05:30 EDT (07:30 / 09:30 UTC) — inside the
    # 02:00-06:00 wall-clock bucket whose nominal key is the non-existent 02:00.
    frame = _minute_frame(_utc(2025, 3, 9, 7, 30), _utc(2025, 3, 9, 9, 30), interval="2h")
    groups = _apply(bucket_key_expr("4h", SESSIONS[InstrumentType.FUTURES_CME]), frame)

    assert groups.null_count() == 0
    assert groups.n_unique() == 1
    assert groups.unique().to_list() == [_utc(2025, 3, 9, 7, 0)]  # 03:00 EDT


def test_fall_back_repeated_hour_merges_into_one_wall_clock_bucket():
    """On fall-back the 01:00-01:59 local hour occurs twice; wall-clock
    bucketing merges both occurrences into one 1h bucket keyed at the first
    01:00 EDT instant (05:00 UTC)."""
    frame = _minute_frame(_utc(2025, 11, 2, 5, 30), _utc(2025, 11, 2, 6, 30), interval="1h")
    # 05:30 UTC = 01:30 EDT (first pass); 06:30 UTC = 01:30 EST (second pass)
    groups = _apply(bucket_key_expr("1h", SESSIONS[InstrumentType.FUTURES_CME]), frame)

    assert groups.n_unique() == 1
    assert groups.unique().to_list() == [_utc(2025, 11, 2, 5, 0)]  # 01:00 EDT, earliest


# ---------------------------------------------------------------------------
# (c) unknown timeframe raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("session", [None, SESSIONS[InstrumentType.EQUITY_US]])
def test_unknown_timeframe_raises(session):
    with pytest.raises(ValueError, match="timeframe"):
        bucket_key_expr("bogus", session)
