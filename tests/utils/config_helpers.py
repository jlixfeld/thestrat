"""Test configuration helper functions.

Provides helper functions to create common configuration patterns used in tests,
reducing boilerplate and improving test readability.
"""

from thestrat.schemas import AggregationConfig, IndicatorsConfig


def create_aggregation_config(**overrides):
    """Create AggregationConfig with sensible test defaults.

    Provides a base configuration suitable for most tests, defaulting to 1-hour
    timeframe with equities asset class settings. Pydantic validation ensures
    all parameters are properly typed and asset class defaults are applied.

    Args:
        **overrides: Any parameters to override defaults. Common overrides:
            - target_timeframes: List of timeframes (e.g., ["5min", "15min"])
            - asset_class: "equities", "crypto", or "fx"
            - timezone: Timezone string (ignored for crypto/fx, forced to UTC)
            - hour_boundary: Boolean for hour alignment behavior

    Returns:
        AggregationConfig: Fully validated Pydantic configuration instance
        with asset class defaults applied via model validators.

    Note:
        Asset class validation automatically applies appropriate defaults:
        - crypto/fx: Forces UTC timezone, enables hour boundary
        - equities: Respects specified timezone, uses market-based defaults

    Examples:
        # Basic config with 1h timeframe, equities defaults
        config = create_aggregation_config()

        # Multi-timeframe crypto configuration
        config = create_aggregation_config(
            target_timeframes=["5min", "15min"],
            asset_class="crypto"
        )

        # Custom equity market timezone
        config = create_aggregation_config(
            asset_class="equities",
            timezone="US/Pacific"
        )
    """
    defaults = {"target_timeframes": ["1h"]}
    return AggregationConfig(**{**defaults, **overrides})


def create_equity_aggregation_config(**overrides):
    """Create AggregationConfig for equities with common settings.

    Pre-configured for US equity market testing with US/Eastern timezone
    and appropriate market hour handling. Ideal for testing traditional
    market scenarios with session gaps and timezone considerations.

    Args:
        **overrides: Any parameters to override defaults. Common overrides:
            - target_timeframes: Default is ["1h"]
            - timezone: Default is "US/Eastern", can be any valid timezone
            - hour_boundary: Default follows equities asset class behavior
            - session_start: Market session start time

    Returns:
        AggregationConfig: Configured instance optimized for equity testing
        with proper timezone handling and market hour awareness.

    Use Cases:
        - Testing market hour aggregation logic
        - DST transition validation
        - Multi-timezone equity scenarios

    Examples:
        # Standard US equity market config
        config = create_equity_aggregation_config()

        # West coast equity testing
        config = create_equity_aggregation_config(timezone="US/Pacific")

        # Intraday equity aggregation
        config = create_equity_aggregation_config(
            target_timeframes=["5min", "15min"]
        )
    """
    defaults = {"target_timeframes": ["1h"], "asset_class": "equities", "timezone": "US/Eastern"}
    return AggregationConfig(**{**defaults, **overrides})


def create_crypto_aggregation_config(**overrides):
    """Create AggregationConfig for crypto with common settings.

    Pre-configured for cryptocurrency testing with UTC timezone enforcement
    and 24/7 trading behavior. The asset class validator automatically forces
    UTC timezone regardless of any timezone override provided.

    Args:
        **overrides: Any parameters to override defaults. Common overrides:
            - target_timeframes: Default is ["1h"]
            - hour_boundary: Forced to True for crypto by validator
            - session_start: Typically "00:00" for 24/7 markets
            Note: timezone parameter is ignored - always forced to UTC

    Returns:
        AggregationConfig: Configured instance for crypto with UTC timezone
        and 24/7 market behavior enforced by Pydantic validators.

    Validation Behavior:
        - timezone: Always forced to "UTC" regardless of input
        - hour_boundary: Automatically set to True for proper alignment
        - Supports continuous trading patterns without session gaps

    Examples:
        # Standard crypto config (UTC enforced)
        config = create_crypto_aggregation_config()

        # High-frequency crypto testing
        config = create_crypto_aggregation_config(
            target_timeframes=["1min", "5min"]
        )

        # Multi-timeframe crypto analysis
        config = create_crypto_aggregation_config(
            target_timeframes=["15min", "1h", "4h"]
        )
    """
    defaults = {
        "target_timeframes": ["1h"],
        "asset_class": "crypto",
        # timezone will be forced to UTC by the validator
    }
    return AggregationConfig(**{**defaults, **overrides})


def create_fx_aggregation_config(**overrides):
    """Create AggregationConfig for forex with common settings.

    Pre-configured for foreign exchange testing with UTC timezone enforcement
    and appropriate FX market behavior. Like crypto, the asset class validator
    automatically forces UTC timezone for global FX market consistency.

    Args:
        **overrides: Any parameters to override defaults. Common overrides:
            - target_timeframes: Default is ["1h"]
            - hour_boundary: Forced to True for FX by validator
            - session_start: FX market opening time
            Note: timezone parameter is ignored - always forced to UTC

    Returns:
        AggregationConfig: Configured instance for FX with UTC timezone
        and proper global market behavior enforced by validators.

    Validation Behavior:
        - timezone: Always forced to "UTC" regardless of input
        - hour_boundary: Automatically set to True for market alignment
        - Handles weekend gaps typical in FX markets

    Examples:
        # Standard FX config (UTC enforced)
        config = create_fx_aggregation_config()

        # FX scalping timeframes
        config = create_fx_aggregation_config(
            target_timeframes=["1min", "5min", "15min"]
        )

        # Daily FX analysis
        config = create_fx_aggregation_config(
            target_timeframes=["1h", "4h", "1d"]
        )
    """
    defaults = {
        "target_timeframes": ["1h"],
        "asset_class": "fx",
        # timezone will be forced to UTC by the validator
    }
    return AggregationConfig(**{**defaults, **overrides})


def create_indicators_config(**overrides):
    """Create IndicatorsConfig with sensible test defaults.

    Provides a minimal indicators configuration suitable for basic testing.
    Uses empty defaults to allow Pydantic validators to apply appropriate
    schema defaults for swing detection, gap analysis, and other indicators.

    Args:
        **overrides: Any parameters to override defaults. Common overrides:
            - timeframe_configs: Dict or list of TimeframeItemConfig objects
            - swing_points: SwingPointsConfig for swing detection settings
            - gap_detection: GapDetectionConfig for gap analysis
            - signal_generation: Signal processing configuration

    Returns:
        IndicatorsConfig: Configured Pydantic instance with schema defaults
        applied automatically for any unspecified indicator parameters.

    Schema Behavior:
        - Empty defaults allow Pydantic validators to apply proper defaults
        - Indicator parameters get schema defaults (swing_window=5, etc.)
        - Timeframe configurations support both dict and object patterns

    Examples:
        # Minimal config with schema defaults
        config = create_indicators_config()

        # Specific timeframe configuration
        from thestrat.schemas import TimeframeItemConfig
        config = create_indicators_config(
            timeframe_configs=[
                TimeframeItemConfig(timeframes=["5min", "15min"])
            ]
        )

        # Custom swing detection parameters
        from thestrat.schemas import SwingPointsConfig
        config = create_indicators_config(
            swing_points=SwingPointsConfig(
                window=10,
                threshold=2.0
            )
        )
    """
    defaults = {}
    return IndicatorsConfig(**{**defaults, **overrides})
