"""
Unit tests for TheStrat Factory component with Pydantic schema validation.

Tests factory pattern for component creation using validated Pydantic models.
"""

import pytest

from thestrat.aggregation import Aggregation
from thestrat.factory import Factory
from thestrat.indicators import Indicators
from thestrat.schemas import (
    AggregationConfig,
    FactoryConfig,
    GapDetectionConfig,
    IndicatorsConfig,
    SwingPointsConfig,
    TimeframeItemConfig,
)


@pytest.mark.unit
class TestCreateAggregation:
    """Test cases for create_aggregation factory method with Pydantic configs."""

    def test_create_aggregation_minimal_config(self):
        """Test creating aggregation with minimal Pydantic configuration."""
        config = AggregationConfig(target_timeframes=["1h"])
        agg = Factory.create_aggregation(config)

        assert isinstance(agg, Aggregation)
        assert agg.target_timeframes == ["1h"]
        assert agg.asset_class == "equities"  # Default

    def test_create_aggregation_complete_config(self):
        """Test creating aggregation with complete Pydantic configuration."""
        config = AggregationConfig(
            target_timeframes=["5min", "1h"],
            asset_class="crypto",
            timezone="UTC",
            hour_boundary=False,
            session_start="00:00",
        )
        agg = Factory.create_aggregation(config)

        assert isinstance(agg, Aggregation)
        assert agg.target_timeframes == ["5min", "1h"]
        assert agg.asset_class == "crypto"
        assert agg.timezone == "UTC"
        assert agg.hour_boundary is False
        assert agg.session_start == "00:00"

    def test_create_aggregation_all_asset_classes(self):
        """Test creating aggregation with all valid asset classes."""
        valid_classes = ["crypto", "equities", "fx"]

        for asset_class in valid_classes:
            config = AggregationConfig(target_timeframes=["1h"], asset_class=asset_class)
            agg = Factory.create_aggregation(config)

            assert isinstance(agg, Aggregation)
            assert agg.asset_class == asset_class

    def test_create_aggregation_multiple_timeframes(self):
        """Test creating aggregation with multiple timeframes."""
        config = AggregationConfig(target_timeframes=["5min", "15min", "1h", "4h", "1d"])
        agg = Factory.create_aggregation(config)

        assert isinstance(agg, Aggregation)
        assert agg.target_timeframes == ["5min", "15min", "1h", "4h", "1d"]


@pytest.mark.unit
class TestCreateIndicators:
    """Test cases for create_indicators factory method with Pydantic configs."""

    def test_create_indicators_minimal_config(self):
        """Test creating indicators with minimal Pydantic configuration."""
        config = IndicatorsConfig(timeframe_configs=[TimeframeItemConfig(timeframes=["all"])])
        indicators = Factory.create_indicators(config)

        assert isinstance(indicators, Indicators)
        assert len(indicators.config.timeframe_configs) == 1
        assert indicators.config.timeframe_configs[0].timeframes == ["all"]
        # Check that config is properly stored (defaults are now handled in processing)
        assert indicators.config.timeframe_configs[0].swing_points is None

    def test_create_indicators_complete_config(self):
        """Test creating indicators with complete Pydantic configuration."""
        config = IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["5min", "15min"],
                    swing_points=SwingPointsConfig(window=7, threshold=3.0),
                    gap_detection=GapDetectionConfig(threshold=0.002),
                )
            ]
        )
        indicators = Factory.create_indicators(config)

        assert isinstance(indicators, Indicators)
        assert len(indicators.config.timeframe_configs) == 1

        tf_config = indicators.config.timeframe_configs[0]
        assert tf_config.timeframes == ["5min", "15min"]
        assert tf_config.swing_points is not None
        assert tf_config.swing_points.window == 7
        assert tf_config.swing_points.threshold == 3.0
        assert tf_config.gap_detection is not None
        assert tf_config.gap_detection.threshold == 0.002

    def test_create_indicators_multiple_timeframe_configs(self):
        """Test creating indicators with multiple timeframe configurations."""
        config = IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(timeframes=["5min"], swing_points=SwingPointsConfig(window=5, threshold=2.0)),
                TimeframeItemConfig(timeframes=["1h"], swing_points=SwingPointsConfig(window=7, threshold=5.0)),
                TimeframeItemConfig(timeframes=["1d"], gap_detection=GapDetectionConfig(threshold=0.01)),
            ]
        )
        indicators = Factory.create_indicators(config)

        assert isinstance(indicators, Indicators)
        assert len(indicators.config.timeframe_configs) == 3

        # Check first config
        assert indicators.config.timeframe_configs[0].timeframes == ["5min"]
        assert indicators.config.timeframe_configs[0].swing_points is not None
        assert indicators.config.timeframe_configs[0].swing_points.threshold == 2.0

        # Check second config
        assert indicators.config.timeframe_configs[1].timeframes == ["1h"]
        assert indicators.config.timeframe_configs[1].swing_points is not None
        assert indicators.config.timeframe_configs[1].swing_points.window == 7

        # Check third config
        assert indicators.config.timeframe_configs[2].timeframes == ["1d"]
        assert indicators.config.timeframe_configs[2].gap_detection is not None
        assert indicators.config.timeframe_configs[2].gap_detection.threshold == 0.01

    def test_create_indicators_optional_configs(self):
        """Test creating indicators with optional swing_points and gap_detection."""
        config = IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(timeframes=["5min"], swing_points=SwingPointsConfig(window=10, threshold=1.5)),
                TimeframeItemConfig(timeframes=["1h"], gap_detection=GapDetectionConfig(threshold=0.005)),
                TimeframeItemConfig(
                    timeframes=["1d"]
                    # No swing_points or gap_detection
                ),
            ]
        )
        indicators = Factory.create_indicators(config)

        assert isinstance(indicators, Indicators)
        assert len(indicators.config.timeframe_configs) == 3

        # First config has swing_points
        assert indicators.config.timeframe_configs[0].swing_points is not None
        assert indicators.config.timeframe_configs[0].gap_detection is None

        # Second config has gap_detection
        assert indicators.config.timeframe_configs[1].swing_points is None
        assert indicators.config.timeframe_configs[1].gap_detection is not None

        # Third config has neither
        assert indicators.config.timeframe_configs[2].swing_points is None
        assert indicators.config.timeframe_configs[2].gap_detection is None


@pytest.mark.unit
class TestCreateAll:
    """Test cases for create_all factory method with Pydantic configs."""

    def test_create_all_minimal_config(self):
        """Test creating all components with minimal Pydantic configuration."""
        config = FactoryConfig(
            aggregation=AggregationConfig(target_timeframes=["1h"]),
            indicators=IndicatorsConfig(timeframe_configs=[TimeframeItemConfig(timeframes=["all"])]),
        )
        components = Factory.create_all(config)

        assert isinstance(components, dict)
        assert "aggregation" in components
        assert "indicators" in components
        assert isinstance(components["aggregation"], Aggregation)
        assert isinstance(components["indicators"], Indicators)

        # Check aggregation
        assert components["aggregation"].target_timeframes == ["1h"]
        assert components["aggregation"].asset_class == "equities"

        # Check indicators
        assert len(components["indicators"].config.timeframe_configs) == 1
        assert components["indicators"].config.timeframe_configs[0].timeframes == ["all"]

    def test_create_all_complete_config(self):
        """Test creating all components with complete Pydantic configuration."""
        config = FactoryConfig(
            aggregation=AggregationConfig(
                target_timeframes=["5min", "1h"], asset_class="crypto", timezone="UTC", hour_boundary=False
            ),
            indicators=IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["5min", "1h"],
                        swing_points=SwingPointsConfig(window=7, threshold=3.0),
                        gap_detection=GapDetectionConfig(threshold=0.002),
                    )
                ]
            ),
        )
        components = Factory.create_all(config)

        # Check aggregation component
        agg = components["aggregation"]
        assert agg.target_timeframes == ["5min", "1h"]
        assert agg.asset_class == "crypto"
        assert agg.timezone == "UTC"
        assert agg.hour_boundary is False

        # Check indicators component
        indicators = components["indicators"]
        tf_config = indicators.config.timeframe_configs[0]
        assert tf_config.timeframes == ["5min", "1h"]
        assert tf_config.swing_points is not None
        assert tf_config.swing_points.window == 7
        assert tf_config.swing_points.threshold == 3.0
        assert tf_config.gap_detection is not None
        assert tf_config.gap_detection.threshold == 0.002

    def test_create_all_complex_config(self):
        """Test creating all components with complex multi-timeframe configuration."""
        config = FactoryConfig(
            aggregation=AggregationConfig(
                target_timeframes=["5min", "15min", "1h", "4h", "1d"],
                asset_class="equities",
                timezone="US/Eastern",
                session_start="09:30",
            ),
            indicators=IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["5min", "15min"], swing_points=SwingPointsConfig(window=5, threshold=2.0)
                    ),
                    TimeframeItemConfig(
                        timeframes=["1h", "4h"],
                        swing_points=SwingPointsConfig(window=7, threshold=3.5),
                        gap_detection=GapDetectionConfig(threshold=0.01),
                    ),
                    TimeframeItemConfig(timeframes=["1d"], swing_points=SwingPointsConfig(window=10, threshold=5.0)),
                ]
            ),
        )
        components = Factory.create_all(config)

        # Verify aggregation
        agg = components["aggregation"]
        assert len(agg.target_timeframes) == 5
        assert agg.asset_class == "equities"
        assert agg.timezone == "US/Eastern"
        assert agg.session_start == "09:30"

        # Verify indicators
        indicators = components["indicators"]
        assert len(indicators.config.timeframe_configs) == 3

        # Check each timeframe config
        assert indicators.config.timeframe_configs[0].timeframes == ["5min", "15min"]
        assert indicators.config.timeframe_configs[0].swing_points is not None
        assert indicators.config.timeframe_configs[0].swing_points.threshold == 2.0

        assert indicators.config.timeframe_configs[1].timeframes == ["1h", "4h"]
        assert indicators.config.timeframe_configs[1].swing_points is not None
        assert indicators.config.timeframe_configs[1].swing_points.threshold == 3.5
        assert indicators.config.timeframe_configs[1].gap_detection is not None
        assert indicators.config.timeframe_configs[1].gap_detection.threshold == 0.01

        assert indicators.config.timeframe_configs[2].timeframes == ["1d"]
        assert indicators.config.timeframe_configs[2].swing_points is not None
        assert indicators.config.timeframe_configs[2].swing_points.window == 10


@pytest.mark.unit
class TestUtilityMethods:
    """Test cases for utility methods (unchanged from original)."""

    def test_get_supported_asset_classes(self):
        """Test getting supported asset classes."""
        classes = Factory.get_supported_asset_classes()

        expected_classes = ["crypto", "equities", "fx"]
        assert isinstance(classes, list)
        assert set(classes) == set(expected_classes)

    def test_get_asset_class_config_valid(self):
        """Test getting asset class configuration for valid classes."""
        for asset_class in ["crypto", "equities", "fx"]:
            config = Factory.get_asset_class_config(asset_class)

            assert isinstance(config, dict)
            assert "timezone" in config
            assert "session_start" in config

    def test_get_asset_class_config_invalid(self):
        """Test getting asset class configuration for invalid class."""
        with pytest.raises(ValueError, match="Unsupported asset class"):
            Factory.get_asset_class_config("invalid")

    def test_validate_timeframe_format_valid(self):
        """Test timeframe format validation for valid formats."""
        valid_timeframes = [
            "1m",
            "5min",
            "15min",
            "30m",
            "1h",
            "2h",
            "4h",
            "8h",
            "1h",
            "2H",
            "4H",
            "6H",
            "1d",
            "1d",
            "7D",
            "14D",
            "1w",
            "1W",
            "2W",
            "1M",
            "3M",
            "6M",
        ]

        for timeframe in valid_timeframes:
            assert Factory.validate_timeframe_format(timeframe) is True

    def test_validate_timeframe_format_invalid(self):
        """Test timeframe format validation for invalid formats."""
        invalid_timeframes = [
            "invalid",
            "1",
            "m",
            "1x",
            "1.5m",
            "",
            "10mins",  # Not in supported units
        ]

        for timeframe in invalid_timeframes:
            assert Factory.validate_timeframe_format(timeframe) is False

    def test_validate_timeframe_format_supported_units(self):
        """Test that supported timeframes and polars-style variations work."""
        # Test exact supported timeframes from TIMEFRAME_TO_POLARS
        supported_timeframes = ["1min", "5min", "15min", "30min", "1h", "4h", "1d", "1w", "1m", "1q", "1y"]
        for timeframe in supported_timeframes:
            assert Factory.validate_timeframe_format(timeframe) is True

        # Test polars-style patterns that should work
        polars_patterns = ["1m", "5m", "15m", "30m", "2h", "3h", "2d", "3d", "2w", "1mo", "3mo", "2y"]
        for timeframe in polars_patterns:
            assert Factory.validate_timeframe_format(timeframe) is True

        # Test unsupported patterns that should fail
        invalid_patterns = ["1minutes", "1hour", "1hours", "1day", "1days", "1week", "1weeks", "1month", "1months"]
        for timeframe in invalid_patterns:
            assert Factory.validate_timeframe_format(timeframe) is False


@pytest.mark.unit
class TestPydanticIntegration:
    """Test cases for Pydantic model integration with Factory."""

    def test_config_reusability(self):
        """Test that Pydantic configs can be reused across Factory calls."""
        agg_config = AggregationConfig(target_timeframes=["5min", "1h"], asset_class="crypto")

        # Create aggregation multiple times with same config
        agg1 = Factory.create_aggregation(agg_config)
        agg2 = Factory.create_aggregation(agg_config)

        # Both should be identical but separate instances
        assert agg1.target_timeframes == agg2.target_timeframes
        assert agg1.asset_class == agg2.asset_class
        assert agg1 is not agg2  # Different instances

    def test_config_modification_safety(self):
        """Test that modifying factory output doesn't affect original config."""
        config = AggregationConfig(target_timeframes=["5min"], asset_class="crypto")

        agg = Factory.create_aggregation(config)

        # Modify the aggregation
        agg.target_timeframes.append("1h")

        # Original config should be unchanged
        assert config.target_timeframes == ["5min"]

        # Create new aggregation from same config
        agg2 = Factory.create_aggregation(config)
        assert agg2.target_timeframes == ["5min"]  # Should still be original

    def test_nested_model_conversion(self):
        """Test proper conversion of nested Pydantic models to dicts."""
        config = IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["5min"],
                    swing_points=SwingPointsConfig(window=7, threshold=3.0),
                    gap_detection=GapDetectionConfig(threshold=0.002),
                )
            ]
        )

        indicators = Factory.create_indicators(config)

        # Verify the nested structure is properly converted
        tf_config = indicators.config.timeframe_configs[0]
        assert isinstance(tf_config.swing_points, SwingPointsConfig)
        assert isinstance(tf_config.gap_detection, GapDetectionConfig)
        assert tf_config.swing_points.window == 7
        assert tf_config.gap_detection.threshold == 0.002
