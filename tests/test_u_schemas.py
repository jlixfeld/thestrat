"""
Test module for schema validation functionality.
This module tests schema validation and related features.
"""

from thestrat.schemas import (
    IndicatorSchema,
    SwingPointsConfig,
)


class TestIndicatorSchemaClassMethods:
    """Test IndicatorSchema class methods."""

    def test_get_column_descriptions(self):
        """Test get_column_descriptions returns proper descriptions."""
        descriptions = IndicatorSchema.get_column_descriptions()

        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0

        # Check some known descriptions
        assert "timestamp" in descriptions
        assert "open" in descriptions
        assert "higher_high" in descriptions
        assert isinstance(descriptions["timestamp"], str)

    def test_get_polars_dtypes(self):
        """Test get_polars_dtypes returns Polars type mappings."""
        polars_types = IndicatorSchema.get_polars_dtypes()

        assert isinstance(polars_types, dict)
        assert len(polars_types) > 0

        # Check some expected mappings
        from polars import Datetime, Float64, Int32, String

        assert "timestamp" in polars_types
        assert polars_types["timestamp"] == Datetime
        assert polars_types["open"] == Float64
        assert polars_types["gapper"] == Int32
        assert polars_types["signal"] == String

    def test_validate_dataframe(self):
        """Test DataFrame column validation function."""
        from datetime import datetime

        from polars import DataFrame, Datetime, Float64, String

        # Create a simple DataFrame with required input columns
        data = {
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "symbol": ["AAPL"],
            "volume": [1000.0],
            "timeframe": ["5min"],
        }
        df = DataFrame(
            data,
            schema={
                "timestamp": Datetime,
                "open": Float64,
                "high": Float64,
                "low": Float64,
                "close": Float64,
                "symbol": String,
                "volume": Float64,
                "timeframe": String,
            },
        )

        result = IndicatorSchema.validate_dataframe(df)

        assert result["valid"] is True
        assert len(result["missing_required"]) == 0
        assert "timestamp" in result["required_fields"]
        assert result["df_type"] == "polars"

    def test_get_column_categories(self):
        """Test get_column_categories returns proper categorization."""
        categories = IndicatorSchema.get_column_categories()

        assert isinstance(categories, dict)
        assert len(categories) > 0

        # Check expected categories exist
        expected_categories = [
            "base_ohlc",
            "price_analysis",
            "gap_detection",
            "market_structure",
            "thestrat_patterns",
            "signals",
            "special_patterns",
            "mother_bar",
        ]

        for category in expected_categories:
            assert category in categories
            assert isinstance(categories[category], list)
            assert len(categories[category]) > 0

        # Check some expected column placements
        assert "timestamp" in categories["base_ohlc"]
        assert "higher_high" in categories["market_structure"]
        assert "continuity" in categories["thestrat_patterns"]
        assert "signal" in categories["signals"]


class TestIndicatorSchemaValidation:
    """Test suite for validating IndicatorSchema alignment with actual processing output.

    These tests ensure that the IndicatorSchema accurately reflects the actual output columns
    from the indicators processing pipeline, preventing schema-output mismatches.
    """

    def create_sample_market_data(self, num_rows: int = 50):
        """Create realistic sample market data for testing."""
        import random
        from datetime import datetime, timedelta

        from polars import DataFrame

        random.seed(42)  # For reproducible results

        timestamps = [datetime(2023, 1, 1) + timedelta(minutes=5 * i) for i in range(num_rows)]

        # Generate realistic OHLC data with some volatility
        base_price = 100.0
        data = []

        for i, timestamp in enumerate(timestamps):
            # Add some trend and volatility
            trend = i * 0.1
            volatility = random.uniform(-2, 2)

            open_price = base_price + trend + volatility

            # Generate high/low around open with some range
            range_size = random.uniform(0.5, 3.0)
            high = open_price + random.uniform(0, range_size)
            low = open_price - random.uniform(0, range_size)

            # Close somewhere in the range
            close = random.uniform(low, high)

            # Volume
            volume = random.uniform(1000, 5000)

            data.append(
                {
                    "timestamp": timestamp,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": round(volume, 0),
                    "symbol": "TEST",
                }
            )

        return DataFrame(data)

    def get_schema_field_categories(self):
        """Extract field names from IndicatorSchema categorized by input/output type."""
        input_only_fields = set()
        output_only_fields = set()
        input_output_fields = set()

        for field_name, field_info in IndicatorSchema.model_fields.items():
            json_extra = getattr(field_info, "json_schema_extra", {}) or {}
            is_input = json_extra.get("input", False)
            is_output = json_extra.get("output", False)

            if is_input and is_output:
                input_output_fields.add(field_name)
            elif is_input:
                input_only_fields.add(field_name)
            elif is_output:
                output_only_fields.add(field_name)

        return {
            "input_only": input_only_fields,
            "output_only": output_only_fields,
            "input_output": input_output_fields,
        }

    def test_schema_output_alignment(self):
        """Test that all schema output fields are present in actual processing output."""
        from thestrat.indicators import Indicators
        from thestrat.schemas import GapDetectionConfig, IndicatorsConfig, TimeframeItemConfig

        sample_data = self.create_sample_market_data(50)

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=5, threshold=5.0),
                        gap_detection=GapDetectionConfig(threshold=0.001),
                    )
                ]
            )
        )
        processed_data = indicators.process(sample_data)

        actual_columns = set(processed_data.columns)
        schema_fields = self.get_schema_field_categories()
        expected_output_columns = schema_fields["output_only"] | schema_fields["input_output"]

        # Find missing columns (schema defines but output doesn't have)
        missing_from_output = expected_output_columns - actual_columns

        assert len(missing_from_output) == 0, (
            f"IndicatorSchema defines {len(missing_from_output)} output columns that are missing from actual output:\n"
            f"{sorted(missing_from_output)}\n"
            f"These columns are likely temporary calculation columns that should be removed from schema."
        )

    def test_no_unexpected_output_columns(self):
        """Test that output doesn't contain columns not defined in schema."""
        from thestrat.indicators import Indicators
        from thestrat.schemas import GapDetectionConfig, IndicatorsConfig, TimeframeItemConfig

        sample_data = self.create_sample_market_data(50)

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=5, threshold=5.0),
                        gap_detection=GapDetectionConfig(threshold=0.001),
                    )
                ]
            )
        )
        processed_data = indicators.process(sample_data)

        actual_columns = set(processed_data.columns)
        schema_fields = self.get_schema_field_categories()
        all_schema_columns = schema_fields["input_only"] | schema_fields["output_only"] | schema_fields["input_output"]

        # Find extra columns (output has but schema doesn't define)
        extra_in_output = actual_columns - all_schema_columns

        assert len(extra_in_output) == 0, (
            f"Processing output contains {len(extra_in_output)} columns not defined in IndicatorSchema:\n"
            f"{sorted(extra_in_output)}\n"
            f"These columns should be added to IndicatorSchema or are unexpected outputs."
        )

    def test_motherbar_problems_present(self):
        """Test that motherbar_problems column is present (regression test for the original issue)."""
        from polars import Boolean

        from thestrat.indicators import Indicators
        from thestrat.schemas import GapDetectionConfig, IndicatorsConfig, TimeframeItemConfig

        sample_data = self.create_sample_market_data(50)

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=5, threshold=5.0),
                        gap_detection=GapDetectionConfig(threshold=0.001),
                    )
                ]
            )
        )
        processed_data = indicators.process(sample_data)

        assert "motherbar_problems" in processed_data.columns, (
            "motherbar_problems column is missing from output. This is the key column for mother bar analysis."
        )

        # Verify it has the correct data type (boolean)
        assert processed_data["motherbar_problems"].dtype == Boolean, (
            f"motherbar_problems should be Boolean type, got {processed_data['motherbar_problems'].dtype}"
        )

    def test_temporary_columns_not_present(self):
        """Test that known temporary columns are not present in output (regression test)."""
        from thestrat.indicators import Indicators
        from thestrat.schemas import GapDetectionConfig, IndicatorsConfig, TimeframeItemConfig

        sample_data = self.create_sample_market_data(50)

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=5, threshold=5.0),
                        gap_detection=GapDetectionConfig(threshold=0.001),
                    )
                ]
            )
        )
        processed_data = indicators.process(sample_data)

        # These are the columns that were incorrectly in the schema before the fix
        temporary_columns = {
            "is_mother_bar",
            "active_mother_high",
            "active_mother_low",
            "in_force_base",
            "pattern_2bar",
            "pattern_3bar",
        }

        actual_columns = set(processed_data.columns)
        found_temporary = temporary_columns & actual_columns

        assert len(found_temporary) == 0, (
            f"Found temporary columns in output that should have been dropped: {sorted(found_temporary)}\n"
            f"These columns should be dropped during processing and not appear in final output."
        )
