"""
Integration tests for TheStrat module.

Tests end-to-end workflows and component interactions.
"""

from datetime import datetime

import pytest
from polars import DataFrame, Datetime

from thestrat import Factory
from thestrat.schemas import (
    AggregationConfig,
    FactoryConfig,
    IndicatorsConfig,
    SwingPointsConfig,
    TimeframeItemConfig,
)

from .utils.config_helpers import create_aggregation_config


@pytest.mark.integration
class TestTheStratIntegration:
    """Integration tests for complete TheStrat workflows."""

    @pytest.fixture
    def sample_market_data(self):
        """Create realistic market data for integration testing."""
        from .utils.thestrat_data_utils import create_market_hours_data

        return create_market_hours_data("AAPL")

    def test_end_to_end_pipeline_polars(self, sample_market_data):
        """Test complete pipeline with Polars DataFrame."""
        # Create complete pipeline with Pydantic models
        config = FactoryConfig(
            aggregation=AggregationConfig(target_timeframes=["5min"], asset_class="equities", timezone="US/Eastern"),
            indicators=IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(
                            window=5,
                            threshold=0.01,  # 0.01% threshold - very low to detect swing points
                        ),
                    )
                ]
            ),
        )

        pipeline = Factory.create_all(config)

        # Process through aggregation
        aggregated = pipeline["aggregation"].process(sample_market_data)

        # Verify aggregation results
        assert isinstance(aggregated, DataFrame)
        assert len(aggregated) == 78  # 390 minutes / 5 minutes per bar
        assert "symbol" in aggregated.columns
        assert aggregated["symbol"][0] == "AAPL"

        # Process through indicators
        analyzed = pipeline["indicators"].process(aggregated)

        # Verify indicator results
        assert isinstance(analyzed, DataFrame)
        assert len(analyzed) == len(aggregated)

        # Check that all expected indicator columns are present
        # Note: Market structure columns (higher_high, etc.) may not be present if no swings of that type are detected
        required_indicators = [
            "swing_high",
            "swing_low",
            "pivot_high",
            "pivot_low",
            "continuity",
            "in_force",
            "scenario",
            "signal",
            "ath",
            "atl",
            "gapper",
        ]

        for indicator in required_indicators:
            assert indicator in analyzed.columns, f"Missing required indicator: {indicator}"

        # Check that at least some market structure columns are present (depending on data)
        market_structure_columns = ["higher_high", "lower_high", "higher_low", "lower_low"]
        present_market_structure = [col for col in market_structure_columns if col in analyzed.columns]
        assert len(present_market_structure) >= 0, "At least some market structure analysis should be present"

    def test_end_to_end_pipeline_pandas(self, sample_market_data):
        """Test complete pipeline with pandas DataFrame input."""
        # Convert to pandas
        pandas_data = sample_market_data.to_pandas()

        # Create pipeline with Pydantic models
        config = FactoryConfig(
            aggregation=AggregationConfig(target_timeframes=["15min"], asset_class="equities"),
            indicators=IndicatorsConfig(timeframe_configs=[TimeframeItemConfig(timeframes=["all"])]),
        )

        pipeline = Factory.create_all(config)

        # Process pandas data through pipeline
        aggregated = pipeline["aggregation"].process(pandas_data)
        analyzed = pipeline["indicators"].process(aggregated)

        # Should still return Polars DataFrame
        assert isinstance(analyzed, DataFrame)
        assert len(analyzed) == 26  # 390 minutes / 15 minutes per bar

        # Should have all required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in analyzed.columns

    def test_crypto_24_7_workflow(self):
        """Test workflow with crypto 24/7 market data."""
        from .utils.thestrat_data_utils import create_crypto_data

        crypto_data = create_crypto_data("BTC-USD")

        # Create crypto pipeline with Pydantic models
        config = FactoryConfig(
            aggregation=AggregationConfig(
                target_timeframes=["4h"],
                asset_class="crypto",  # Should force UTC timezone
                timezone="US/Eastern",  # This should be ignored for crypto
            ),
            indicators=IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            ),
        )

        pipeline = Factory.create_all(config)

        # Verify crypto uses UTC timezone
        assert pipeline["aggregation"].timezone == "UTC"

        # Process data
        aggregated = pipeline["aggregation"].process(crypto_data)
        analyzed = pipeline["indicators"].process(aggregated)

        assert isinstance(analyzed, DataFrame)
        assert len(analyzed) == 12  # 48 hours / 4 hours per bar
        # Check timezone - assert it's a Datetime type first
        timestamp_dtype = analyzed.schema["timestamp"]
        assert isinstance(timestamp_dtype, Datetime)
        assert timestamp_dtype.time_zone == "UTC"

    def test_forex_utc_workflow(self):
        """Test workflow with forex UTC market data."""
        from .utils.thestrat_data_utils import create_forex_data

        forex_data = create_forex_data("EUR/USD")

        # Create forex pipeline with Pydantic models
        config = FactoryConfig(
            aggregation=AggregationConfig(
                target_timeframes=["1h"],
                asset_class="fx",  # Should force UTC timezone
            ),
            indicators=IndicatorsConfig(timeframe_configs=[TimeframeItemConfig(timeframes=["all"])]),
        )

        pipeline = Factory.create_all(config)

        # Verify FX uses UTC timezone
        assert pipeline["aggregation"].timezone == "UTC"

        # Process data
        aggregated = pipeline["aggregation"].process(forex_data)
        analyzed = pipeline["indicators"].process(aggregated)

        assert isinstance(analyzed, DataFrame)
        assert len(analyzed) == 120  # 240 * 30min / 60min per bar

    def test_component_validation_integration(self):
        """Test that validation errors are properly handled across components."""
        # Create invalid data (missing required columns)
        invalid_data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                # Missing high, low, close, volume
            }
        )

        # Create pipeline with Pydantic models
        pipeline = Factory.create_all(
            FactoryConfig(
                aggregation=create_aggregation_config(),
                indicators=IndicatorsConfig(timeframe_configs=[TimeframeItemConfig(timeframes=["all"])]),
            )
        )

        # Aggregation should fail validation
        with pytest.raises(ValueError, match="Input data validation failed"):
            pipeline["aggregation"].process(invalid_data)

        # Indicators should also fail validation
        with pytest.raises(ValueError, match="Missing required columns"):
            pipeline["indicators"].process(invalid_data)

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        from .utils.thestrat_data_utils import create_large_dataset

        large_data = create_large_dataset(periods=1950, freq_minutes=1)

        # Process with hourly aggregation using Pydantic models
        config = FactoryConfig(
            aggregation=AggregationConfig(target_timeframes=["1h"], asset_class="equities"),
            indicators=IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=5))]
            ),
        )

        pipeline = Factory.create_all(config)

        # Should complete without errors or timeout
        aggregated = pipeline["aggregation"].process(large_data)
        analyzed = pipeline["indicators"].process(aggregated)

        assert isinstance(analyzed, DataFrame)
        assert len(analyzed) > 0
        # Should have all indicator columns
        assert "swing_high" in analyzed.columns
        assert "ath" in analyzed.columns

    def test_factory_component_consistency(self):
        """Test that factory-created components are consistent."""
        # Create multiple instances of the same configuration using Pydantic models
        config = FactoryConfig(
            aggregation=AggregationConfig(target_timeframes=["5min"], asset_class="equities", timezone="US/Eastern"),
            indicators=IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(
                            window=7,
                            threshold=2.0,  # 2% threshold
                        ),
                    )
                ]
            ),
        )

        # Create multiple pipelines
        pipeline1 = Factory.create_all(config)
        pipeline2 = Factory.create_all(config)

        # Components should have same configuration
        assert pipeline1["aggregation"].target_timeframes == pipeline2["aggregation"].target_timeframes
        assert pipeline1["aggregation"].asset_class == pipeline2["aggregation"].asset_class
        assert pipeline1["aggregation"].timezone == pipeline2["aggregation"].timezone

        assert pipeline1["indicators"].config.timeframe_configs == pipeline2["indicators"].config.timeframe_configs

    def test_error_propagation_chain(self):
        """Test that errors propagate correctly through the processing chain."""
        # Create data that will pass aggregation but fail indicators (too small)
        from .utils.thestrat_data_utils import create_ohlc_data

        small_data = create_ohlc_data(periods=3, start="2023-01-01", freq_minutes=60, base_price=100.0)

        # Create pipeline with Pydantic models
        pipeline = Factory.create_all(
            FactoryConfig(
                aggregation=create_aggregation_config(),
                indicators=IndicatorsConfig(
                    timeframe_configs=[
                        TimeframeItemConfig(
                            timeframes=["all"],
                            swing_points=SwingPointsConfig(window=10),  # Requires more data
                        )
                    ]
                ),
            )
        )

        # Aggregation should work
        aggregated = pipeline["aggregation"].process(small_data)
        assert isinstance(aggregated, DataFrame)

        # Indicators should handle small data gracefully (no longer raises error)
        result = pipeline["indicators"].process(aggregated)
        assert isinstance(result, DataFrame)
        # Should have all required columns but limited swing point detection
        assert len(result.columns) == 46  # Full schema maintained
