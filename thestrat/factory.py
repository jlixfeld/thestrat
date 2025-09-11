"""
Factory pattern for TheStrat component creation and configuration.

This module provides clean factory methods for creating and configuring
TheStrat components with Pydantic schema validation.
"""

from typing import Any

from .aggregation import Aggregation
from .base import Component
from .indicators import Indicators
from .schemas import AggregationConfig, FactoryConfig, IndicatorsConfig


class Factory:
    """
    Factory class for creating TheStrat components.

    Provides static methods for creating individual components and
    class methods for creating complete processing pipelines.
    All methods accept validated Pydantic configuration models.
    """

    @staticmethod
    def create_aggregation(config: AggregationConfig) -> Aggregation:
        """
        Create aggregation component from validated configuration.

        Args:
            config: Validated AggregationConfig containing:
                - `target_timeframes`: Target timeframes list (e.g., ["1H", "5m", "1D"])
                - `asset_class`: Asset class type (crypto, equities, fx, futures, commodities)
                - `timezone`: Timezone override (UTC asset classes always use UTC;
                           non-UTC uses priority: specified > system > asset_class default)
                - `hour_boundary`: Align to hour boundaries (auto-determined if None)
                - `session_start`: Session start time (asset class default if None)

        Returns:
            Configured Aggregation component

        Example:
            >>> from thestrat.schemas import AggregationConfig
            >>> config = AggregationConfig(
            ...     target_timeframes=["5m", "1H"],
            ...     asset_class="crypto",
            ...     timezone="UTC"
            ... )
            >>> aggregation = Factory.create_aggregation(config)
        """
        return Aggregation(
            target_timeframes=config.target_timeframes,
            asset_class=config.asset_class,
            timezone=config.timezone,
            hour_boundary=config.hour_boundary,
            session_start=config.session_start,
        )

    @staticmethod
    def create_indicators(config: IndicatorsConfig) -> Indicators:
        """
        Create indicators component from validated configuration.

        Args:
            config: Validated IndicatorsConfig containing per-timeframe configurations
                using Pydantic models

        Returns:
            Configured Indicators component

        Example:
            >>> from thestrat.schemas import IndicatorsConfig, TimeframeItemConfig, SwingPointsConfig
            >>> config = IndicatorsConfig(
            ...     timeframe_configs=[
            ...         TimeframeItemConfig(
            ...             timeframes=["5m", "15m"],
            ...             swing_points=SwingPointsConfig(window=7, threshold=3.0)
            ...         )
            ...     ]
            ... )
            >>> indicators = Factory.create_indicators(config)
        """
        # Convert Pydantic models to dicts for the Indicators constructor
        timeframe_configs = []
        for tf_config in config.timeframe_configs:
            config_dict = {"timeframes": tf_config.timeframes}

            if tf_config.swing_points is not None:
                config_dict["swing_points"] = tf_config.swing_points.model_dump()

            if tf_config.gap_detection is not None:
                config_dict["gap_detection"] = tf_config.gap_detection.model_dump()

            timeframe_configs.append(config_dict)

        return Indicators(timeframe_configs=timeframe_configs)

    @classmethod
    def create_all(cls, config: FactoryConfig) -> dict[str, Component]:
        """
        Create complete processing pipeline with aggregation and indicators.

        Args:
            config: Validated FactoryConfig containing:
                `aggregation`: AggregationConfig with target timeframes and settings
                `indicators`: IndicatorsConfig with per-timeframe configurations

        Returns:
            Dictionary containing created components:
            {
                "aggregation": Aggregation,
                "indicators": Indicators
            }

        Example:
            >>> from thestrat.schemas import FactoryConfig, AggregationConfig, IndicatorsConfig, TimeframeItemConfig
            >>> config = FactoryConfig(
            ...     aggregation=AggregationConfig(target_timeframes=["5m"]),
            ...     indicators=IndicatorsConfig(
            ...         timeframe_configs=[TimeframeItemConfig(timeframes=["all"])]
            ...     )
            ... )
            >>> components = Factory.create_all(config)
        """
        return {
            "aggregation": cls.create_aggregation(config.aggregation),
            "indicators": cls.create_indicators(config.indicators),
        }

    @staticmethod
    def get_supported_asset_classes() -> list[str]:
        """
        Get list of supported asset classes.

        Returns:
            List of supported asset class strings
        """
        from .schemas import AssetClassConfig

        return list(AssetClassConfig.REGISTRY.keys())

    @staticmethod
    def get_asset_class_config(asset_class: str) -> dict[str, Any]:
        """
        Get default configuration for specific asset class.

        Args:
            asset_class: Asset class type

        Returns:
            Dictionary with default `timezone` and `session_start`

        Raises:
            ValueError: If asset_class is not supported
        """
        from .schemas import AssetClassConfig

        if asset_class not in Factory.get_supported_asset_classes():
            raise ValueError(f"Unsupported asset class: {asset_class}")

        return AssetClassConfig.REGISTRY[asset_class].model_dump()

    @staticmethod
    def validate_timeframe_format(timeframe: str) -> bool:
        """
        Validate timeframe format against supported timeframes.

        Args:
            timeframe: Timeframe string to validate

        Returns:
            True if format is valid, False otherwise
        """
        from .schemas import TimeframeConfig

        return TimeframeConfig.validate_timeframe(timeframe)
