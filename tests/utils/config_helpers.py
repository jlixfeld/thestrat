"""Test configuration helper functions.

Provides helper functions to create common configuration patterns used in tests,
reducing boilerplate and improving test readability.
"""

from thestrat.schemas import AggregationConfig, IndicatorsConfig


def create_aggregation_config(**overrides):
    """Create AggregationConfig with sensible test defaults.

    Args:
        **overrides: Any parameters to override defaults

    Returns:
        AggregationConfig: Configured instance

    Example:
        # Basic config with 1h timeframe
        config = create_aggregation_config()

        # Override specific parameters
        config = create_aggregation_config(target_timeframes=["5min"], asset_class="crypto")
    """
    defaults = {"target_timeframes": ["1h"]}
    return AggregationConfig(**{**defaults, **overrides})


def create_equity_aggregation_config(**overrides):
    """Create AggregationConfig for equities with common settings.

    Args:
        **overrides: Any parameters to override defaults

    Returns:
        AggregationConfig: Configured instance for equities

    Example:
        config = create_equity_aggregation_config()  # US/Eastern timezone
        config = create_equity_aggregation_config(timezone="US/Pacific")
    """
    defaults = {"target_timeframes": ["1h"], "asset_class": "equities", "timezone": "US/Eastern"}
    return AggregationConfig(**{**defaults, **overrides})


def create_crypto_aggregation_config(**overrides):
    """Create AggregationConfig for crypto with common settings.

    Args:
        **overrides: Any parameters to override defaults

    Returns:
        AggregationConfig: Configured instance for crypto (UTC timezone)

    Example:
        config = create_crypto_aggregation_config()  # UTC timezone enforced
        config = create_crypto_aggregation_config(target_timeframes=["15min"])
    """
    defaults = {
        "target_timeframes": ["1h"],
        "asset_class": "crypto",
        # timezone will be forced to UTC by the validator
    }
    return AggregationConfig(**{**defaults, **overrides})


def create_fx_aggregation_config(**overrides):
    """Create AggregationConfig for forex with common settings.

    Args:
        **overrides: Any parameters to override defaults

    Returns:
        AggregationConfig: Configured instance for forex (UTC timezone)

    Example:
        config = create_fx_aggregation_config()  # UTC timezone enforced
        config = create_fx_aggregation_config(target_timeframes=["30min"])
    """
    defaults = {
        "target_timeframes": ["1h"],
        "asset_class": "fx",
        # timezone will be forced to UTC by the validator
    }
    return AggregationConfig(**{**defaults, **overrides})


def create_indicators_config(**overrides):
    """Create IndicatorsConfig with sensible test defaults.

    Args:
        **overrides: Any parameters to override defaults

    Returns:
        IndicatorsConfig: Configured instance

    Example:
        config = create_indicators_config()  # Basic config
        config = create_indicators_config(timeframe_configs={"1h": TimeframeConfig(...)})
    """
    defaults = {}
    return IndicatorsConfig(**{**defaults, **overrides})
