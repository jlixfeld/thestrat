"""Test cases for issue #51: Monthly aggregation DST time-of-day fix."""

import polars as pl
from polars import DataFrame, col

from thestrat import Aggregation
from thestrat.schemas import AggregationConfig


class TestMonthlyDSTTimestamps:
    """Test monthly aggregation timestamp handling across DST transitions (Issue #51)."""

    def test_monthly_timestamps_edt_summer(self):
        """Monthly bars during EDT (summer) should be 09:30 ET = 13:30 UTC."""
        from .utils.thestrat_data_utils import create_timestamp_series

        # Create daily bars at 09:30 ET for July 2025 (EDT period)
        # During EDT, 09:30 ET = 13:30 UTC
        timestamps = create_timestamp_series("2025-07-01 09:30:00", 90, 1440)  # 90 days
        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(90)],
                "high": [100.5 + i * 0.01 for i in range(90)],
                "low": [99.5 + i * 0.01 for i in range(90)],
                "close": [100.2 + i * 0.01 for i in range(90)],
                "volume": [1000] * 90,
                "timeframe": ["1d"] * 90,
                "symbol": ["NVDA"] * 90,
            }
        )

        # Set timestamps to 09:30 ET
        test_data = test_data.with_columns([col("timestamp").dt.replace_time_zone("America/New_York")])

        agg = Aggregation(
            AggregationConfig(target_timeframes=["1m"], asset_class="equities", timezone="America/New_York")
        )
        result = agg.process(test_data)

        # Convert to UTC to verify exact time
        result_utc = result.with_columns([col("timestamp").dt.convert_time_zone("UTC")])

        for idx, row in enumerate(result_utc.iter_rows(named=True)):
            ts_utc = row["timestamp"]
            # During EDT (summer), 09:30 ET = 13:30 UTC
            assert ts_utc.hour == 13, f"Row {idx}: Expected 13:30 UTC for EDT, got {ts_utc.hour}:{ts_utc.minute:02d}"
            assert ts_utc.minute == 30, f"Row {idx}: Expected 13:30 UTC for EDT, got {ts_utc.hour}:{ts_utc.minute:02d}"

    def test_monthly_timestamps_est_winter(self):
        """Monthly bars during EST (winter) should be 09:30 ET = 14:30 UTC."""
        from .utils.thestrat_data_utils import create_timestamp_series

        # Create daily bars at 09:30 ET for January-March 2025 (EST period)
        # During EST, 09:30 ET = 14:30 UTC
        timestamps = create_timestamp_series("2025-01-01 09:30:00", 90, 1440)  # 90 days
        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(90)],
                "high": [100.5 + i * 0.01 for i in range(90)],
                "low": [99.5 + i * 0.01 for i in range(90)],
                "close": [100.2 + i * 0.01 for i in range(90)],
                "volume": [1000] * 90,
                "timeframe": ["1d"] * 90,
                "symbol": ["NVDA"] * 90,
            }
        )

        # Set timestamps to 09:30 ET
        test_data = test_data.with_columns([col("timestamp").dt.replace_time_zone("America/New_York")])

        agg = Aggregation(
            AggregationConfig(target_timeframes=["1m"], asset_class="equities", timezone="America/New_York")
        )
        result = agg.process(test_data)

        # Convert to UTC to verify exact time
        result_utc = result.with_columns([col("timestamp").dt.convert_time_zone("UTC")])

        for idx, row in enumerate(result_utc.iter_rows(named=True)):
            ts_utc = row["timestamp"]
            # During EST (winter), 09:30 ET = 14:30 UTC
            assert ts_utc.hour == 14, f"Row {idx}: Expected 14:30 UTC for EST, got {ts_utc.hour}:{ts_utc.minute:02d}"
            assert ts_utc.minute == 30, f"Row {idx}: Expected 14:30 UTC for EST, got {ts_utc.hour}:{ts_utc.minute:02d}"

    def test_monthly_vs_other_calendar_periods_consistency(self):
        """Monthly timestamps should match quarterly/yearly time-of-day."""
        from .utils.thestrat_data_utils import create_timestamp_series

        # Create daily bars spanning full year at 09:30 ET
        timestamps = create_timestamp_series("2025-01-01 09:30:00", 365, 1440)
        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(365)],
                "high": [100.5 + i * 0.01 for i in range(365)],
                "low": [99.5 + i * 0.01 for i in range(365)],
                "close": [100.2 + i * 0.01 for i in range(365)],
                "volume": [1000] * 365,
                "timeframe": ["1d"] * 365,
                "symbol": ["NVDA"] * 365,
            }
        )

        test_data = test_data.with_columns([col("timestamp").dt.replace_time_zone("America/New_York")])

        # Aggregate to all calendar periods
        agg_all = Aggregation(
            AggregationConfig(
                target_timeframes=["1m", "1q", "1y"], asset_class="equities", timezone="America/New_York"
            )
        )
        result = agg_all.process(test_data)

        # All calendar periods should have same time-of-day (09:30 ET)
        for row in result.iter_rows(named=True):
            ts = row["timestamp"]
            tf = row["timeframe"]
            assert ts.hour == 9, f"{tf}: Expected hour 9, got {ts.hour} for {ts}"
            assert ts.minute == 30, f"{tf}: Expected minute 30, got {ts.minute} for {ts}"

    def test_monthly_timestamps_across_dst_transition(self):
        """Monthly bars should maintain correct time-of-day across DST transitions."""
        from .utils.thestrat_data_utils import create_timestamp_series

        # Create daily bars spanning DST transitions (Nov 2024 - Nov 2025)
        # Nov 2024 (EST) → Mar 2025 DST transition → Jul 2025 (EDT) → Nov 2025 DST transition
        timestamps = create_timestamp_series("2024-11-01 09:30:00", 400, 1440)
        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(400)],
                "high": [100.5 + i * 0.01 for i in range(400)],
                "low": [99.5 + i * 0.01 for i in range(400)],
                "close": [100.2 + i * 0.01 for i in range(400)],
                "volume": [1000] * 400,
                "timeframe": ["1d"] * 400,
                "symbol": ["NVDA"] * 400,
            }
        )

        test_data = test_data.with_columns([col("timestamp").dt.replace_time_zone("America/New_York")])

        agg = Aggregation(
            AggregationConfig(target_timeframes=["1m"], asset_class="equities", timezone="America/New_York")
        )
        result = agg.process(test_data)

        # All monthly bars should be 09:30 ET regardless of DST
        for row in result.iter_rows(named=True):
            ts = row["timestamp"]
            assert ts.hour == 9, f"Expected 09:30 ET, got {ts.hour}:{ts.minute:02d} for {ts.date()}"
            assert ts.minute == 30, f"Expected 09:30 ET, got {ts.hour}:{ts.minute:02d} for {ts.date()}"

    def test_other_calendar_periods_unchanged(self):
        """Verify daily/weekly/quarterly/yearly still work correctly after monthly fix."""
        from .utils.thestrat_data_utils import create_timestamp_series

        # Create daily bars at 09:30 ET
        timestamps = create_timestamp_series("2025-01-01 09:30:00", 400, 1440)
        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(400)],
                "high": [100.5 + i * 0.01 for i in range(400)],
                "low": [99.5 + i * 0.01 for i in range(400)],
                "close": [100.2 + i * 0.01 for i in range(400)],
                "volume": [1000] * 400,
                "timeframe": ["1d"] * 400,
                "symbol": ["NVDA"] * 400,
            }
        )

        test_data = test_data.with_columns([col("timestamp").dt.replace_time_zone("America/New_York")])

        # Test all non-monthly calendar periods
        for timeframe in ["1d", "1w", "1q", "1y"]:
            agg = Aggregation(
                AggregationConfig(target_timeframes=[timeframe], asset_class="equities", timezone="America/New_York")
            )
            result = agg.process(test_data)

            for row in result.iter_rows(named=True):
                ts = row["timestamp"]
                assert ts.hour == 9, f"{timeframe}: Expected 09:30, got {ts.hour}:{ts.minute:02d} for {ts.date()}"
                assert ts.minute == 30, f"{timeframe}: Expected 09:30, got {ts.hour}:{ts.minute:02d} for {ts.date()}"

    def test_monthly_reproduces_issue51_evidence(self):
        """Reproduce exact evidence from issue #51 and verify fix."""
        # Create data that would reproduce the issue described in #51
        # NVDA equity data with session start 09:30 ET

        timestamps = pl.datetime_range(
            pl.datetime(2025, 1, 1, 9, 30), pl.datetime(2025, 11, 30, 9, 30), interval="1d", eager=True
        )

        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * len(timestamps),
                "high": [101.0] * len(timestamps),
                "low": [99.0] * len(timestamps),
                "close": [100.5] * len(timestamps),
                "volume": [1000000] * len(timestamps),
                "timeframe": ["1d"] * len(timestamps),
                "symbol": ["NVDA"] * len(timestamps),
            }
        )

        test_data = test_data.with_columns([col("timestamp").dt.replace_time_zone("America/New_York")])

        agg = Aggregation(
            AggregationConfig(target_timeframes=["1m"], asset_class="equities", timezone="America/New_York")
        )
        result = agg.process(test_data)

        # Convert to UTC to check the exact times mentioned in issue
        result_utc = result.with_columns([col("timestamp").dt.convert_time_zone("UTC")])

        # Check specific months mentioned in issue #51
        for row in result_utc.iter_rows(named=True):
            ts_utc = row["timestamp"]
            month = ts_utc.month

            if month in [11, 10]:  # EDT months (expect 13:30 UTC)
                assert ts_utc.hour == 13, f"{ts_utc.date()}: EDT month should be 13:30 UTC, got {ts_utc}"
                assert ts_utc.minute == 30
            elif month in [2, 3]:  # EST months (expect 14:30 UTC)
                assert ts_utc.hour == 14, f"{ts_utc.date()}: EST month should be 14:30 UTC, got {ts_utc}"
                assert ts_utc.minute == 30
