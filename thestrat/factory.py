"""
Factory pattern for TheStrat component creation and configuration.

This module provides clean factory methods for creating and configuring
TheStrat components with Pydantic schema validation.
"""

from functools import lru_cache
from typing import Any, TypedDict

from .aggregation import Aggregation
from .indicators import Indicators
from .schemas import AggregationConfig, FactoryConfig, IndicatorsConfig


class ComponentDict(TypedDict):
    """Type definition for component dictionary returned by Factory.create_all()."""

    aggregation: Aggregation
    indicators: Indicators


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
        return Aggregation(config)

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
        return Indicators(config)

    @classmethod
    def create_all(cls, config: FactoryConfig) -> ComponentDict:
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
    @lru_cache(maxsize=1)
    def get_supported_timeframes() -> list[str]:
        """
        Get list of all supported timeframes in chronological order.

        Returns a chronologically ordered list of timeframe strings that can be used with
        the aggregation and indicators components. These timeframes are
        validated when creating configurations.

        Returns:
            Chronologically ordered list of supported timeframe strings (shortest to longest)

        Example:
            >>> from thestrat import Factory
            >>> timeframes = Factory.get_supported_timeframes()
            >>> print(timeframes)
            ['1min', '5min', '15min', '30min', '1h', '4h', '6h', '12h', '1d', '1w', '1m', '1q', '1y']

            >>> # Use in configuration
            >>> from thestrat.schemas import AggregationConfig
            >>> available_timeframes = Factory.get_supported_timeframes()
            >>> config = AggregationConfig(
            ...     target_timeframes=[available_timeframes[0], available_timeframes[1]],
            ...     asset_class="equities"
            ... )

        Note:
            The timeframes are ordered by duration (shortest to longest) and follow standard conventions:
            - Minutes: `1min`, `5min`, `15min`, `30min`
            - Hours: `1h`, `4h`, `6h`, `12h`
            - Days/Weeks/Months: `1d`, `1w`, `1m` (month), `1q` (quarter), `1y`
        """
        from .schemas import TimeframeConfig

        # Pre-compute timeframes with their durations for efficient sorting
        timeframes_with_duration = [
            (tf, TimeframeConfig.TIMEFRAME_METADATA.get(tf, {}).get("seconds", float("inf")))
            for tf in TimeframeConfig.TIMEFRAME_TO_POLARS.keys()
        ]

        # Sort by duration and return just the timeframe strings
        return [tf for tf, _ in sorted(timeframes_with_duration, key=lambda x: x[1])]

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
    def get_timeframe_metadata(timeframe: str) -> dict[str, Any]:
        """
        Get detailed metadata for a specific timeframe.

        Provides comprehensive information about a timeframe including its
        category, duration in seconds, description, and typical use cases.
        This is useful for understanding the characteristics and appropriate
        applications of different timeframes.

        Args:
            timeframe: Timeframe string to get metadata for (e.g., "5min", "1h", "1d")

        Returns:
            Dictionary containing:
            - `category`: Timeframe category (e.g., "sub-hourly", "hourly", "daily")
            - `seconds`: Duration in seconds
            - `description`: Human-readable description
            - `typical_use`: Common use cases for this timeframe
            - `data_volume`: Expected data volume level

        Raises:
            ValueError: If timeframe is not supported

        Example:
            >>> from thestrat import Factory
            >>> metadata = Factory.get_timeframe_metadata("1h")
            >>> print(metadata)
            {
                'category': 'hourly',
                'seconds': 3600,
                'description': '1-hour bars for daily structure analysis',
                'typical_use': 'Day trading context, hourly levels',
                'data_volume': 'low'
            }

            >>> # Check if a timeframe is suitable for day trading
            >>> metadata = Factory.get_timeframe_metadata("5min")
            >>> if "day trading" in metadata["typical_use"].lower():
            ...     print(f"5min is suitable for day trading")

            >>> # Get all intraday timeframes
            >>> all_timeframes = Factory.get_supported_timeframes()
            >>> intraday = []
            >>> for tf in all_timeframes:
            ...     meta = Factory.get_timeframe_metadata(tf)
            ...     if meta["category"] in ["sub-hourly", "hourly"]:
            ...         intraday.append(tf)

        See Also:
            - :meth:`get_supported_timeframes`: Get list of all supported timeframes
            - :meth:`validate_timeframe_format`: Validate a timeframe string
        """
        # Use cached helper method to get metadata efficiently
        metadata = Factory._get_cached_timeframe_metadata(timeframe)
        return metadata.copy()  # Return a copy to prevent modification

    @staticmethod
    @lru_cache(maxsize=32)  # Cache up to 32 timeframes (more than we currently have)
    def _get_cached_timeframe_metadata(timeframe: str) -> dict[str, Any]:
        """
        Internal cached method to retrieve timeframe metadata.

        This method caches the actual metadata lookup while the public method
        returns copies to maintain immutability.
        """
        from .schemas import TimeframeConfig

        if timeframe not in TimeframeConfig.TIMEFRAME_TO_POLARS:
            supported = sorted(TimeframeConfig.TIMEFRAME_TO_POLARS.keys())
            raise ValueError(f"Unsupported timeframe: '{timeframe}'. Supported timeframes are: {supported}")

        metadata = TimeframeConfig.TIMEFRAME_METADATA.get(timeframe)
        if metadata is None:
            # Fallback for timeframes without detailed metadata
            return {
                "category": "unknown",
                "seconds": 0,
                "description": f"Timeframe {timeframe}",
                "typical_use": "Various trading strategies",
                "data_volume": "unknown",
            }

        return metadata

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
