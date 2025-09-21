"""
Pydantic schema models for TheStrat configuration validation.

This module provides comprehensive validation schemas that replace all manual validation
logic in the Factory class. Models use Pydantic v2 features for maximum performance,
type safety, and detailed error reporting.
"""

import re
from datetime import datetime
from typing import Any, ClassVar, Literal, Self

import pytz
from polars import Boolean, Datetime, Float64, Int32, String
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AssetClassConfig(BaseModel):
    """Configuration for a specific asset class with comprehensive market metadata."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")

    timezone: str = Field(
        description="Timezone for market hours and timestamp handling",
        examples=["UTC", "US/Eastern", "Europe/London", "Asia/Tokyo"],
        json_schema_extra={"format": "timezone", "validation": "Must be valid pytz timezone"},
    )
    session_start: str = Field(
        description="Market session start time in HH:MM 24-hour format",
        pattern=r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$",
        examples=["09:30", "00:00", "14:30"],
        json_schema_extra={"format": "HH:MM"},
    )
    hour_boundary: bool = Field(
        description="Whether to align aggregation to hour boundaries for hourly+ timeframes",
        json_schema_extra={
            "impact": "True: align to hour boundaries (00:00, 01:00). False: align to session_start",
            "typical_usage": "True for 24/7 markets, False for session-based markets",
        },
    )
    trading_hours: str = Field(
        description="Human-readable description of trading hours",
        examples=["24/7 continuous", "9:30 AM - 4:00 PM ET", "Sunday 5 PM - Friday 5 PM ET"],
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        # Validate timezone using pytz
        try:
            pytz.timezone(v)
        except pytz.UnknownTimeZoneError as err:
            raise ValueError(f"Unknown timezone: {v}") from err
        return v

    @field_validator("session_start")
    @classmethod
    def validate_session_start(cls, v: str) -> str:
        # Basic time format validation (HH:MM)
        time_pattern = r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$"
        if not re.match(time_pattern, v):
            raise ValueError("session_start must be in HH:MM format")
        return v

    # Registry pattern - similar to TimeframeConfig.TIMEFRAME_METADATA
    REGISTRY: ClassVar[dict[str, "AssetClassConfig"]] = {}

    @classmethod
    def get_config(cls, asset_class: str) -> "AssetClassConfig":
        """Get configuration for specific asset class."""
        return cls.REGISTRY.get(asset_class, cls.REGISTRY["equities"])


# Populate the registry after class definition
AssetClassConfig.REGISTRY = {
    "crypto": AssetClassConfig(
        timezone="UTC", session_start="00:00", hour_boundary=True, trading_hours="24/7 continuous trading"
    ),
    "equities": AssetClassConfig(
        timezone="US/Eastern",
        session_start="09:30",
        hour_boundary=False,
        trading_hours="9:30 AM - 4:00 PM ET (regular session)",
    ),
    "fx": AssetClassConfig(
        timezone="UTC", session_start="00:00", hour_boundary=True, trading_hours="Sunday 5 PM - Friday 5 PM ET (24/5)"
    ),
}

# Keep the old ASSET_CLASS_CONFIGS for backward compatibility - can be removed in future
ASSET_CLASS_CONFIGS = AssetClassConfig.REGISTRY


class TimeframeConfig(BaseModel):
    """Configuration and validation for timeframes with comprehensive metadata."""

    # Comprehensive timeframe metadata (includes Polars format mapping)
    TIMEFRAME_METADATA: ClassVar[dict[str, dict[str, Any]]] = {
        "1min": {
            "category": "sub-hourly",
            "seconds": 60,
            "description": "1-minute bars for high-frequency analysis",
            "typical_use": "Scalping, entry timing, tick analysis",
            "data_volume": "very_high",
            "polars_format": "1m",
        },
        "5min": {
            "category": "sub-hourly",
            "seconds": 300,
            "description": "5-minute bars for short-term patterns",
            "typical_use": "Day trading, quick reversals",
            "data_volume": "high",
            "polars_format": "5m",
        },
        "15min": {
            "category": "sub-hourly",
            "seconds": 900,
            "description": "15-minute bars for intraday structure",
            "typical_use": "Swing entries, session analysis",
            "data_volume": "medium",
            "polars_format": "15m",
        },
        "30min": {
            "category": "sub-hourly",
            "seconds": 1800,
            "description": "30-minute bars for session structure",
            "typical_use": "Half-hourly patterns, session transitions",
            "data_volume": "medium",
            "polars_format": "30m",
        },
        "1h": {
            "category": "hourly",
            "seconds": 3600,
            "description": "1-hour bars for daily structure analysis",
            "typical_use": "Day trading context, hourly levels",
            "data_volume": "low",
            "polars_format": "1h",
        },
        "4h": {
            "category": "multi-hourly",
            "seconds": 14400,
            "description": "4-hour bars for swing trading",
            "typical_use": "Swing trading, multi-day holds",
            "data_volume": "very_low",
            "polars_format": "4h",
        },
        "6h": {
            "category": "multi-hourly",
            "seconds": 21600,
            "description": "6-hour bars for extended swing analysis",
            "typical_use": "Position trading context",
            "data_volume": "very_low",
            "polars_format": "6h",
        },
        "12h": {
            "category": "multi-hourly",
            "seconds": 43200,
            "description": "12-hour bars for daily session analysis",
            "typical_use": "Asian/Western session splits",
            "data_volume": "very_low",
            "polars_format": "12h",
        },
        "1d": {
            "category": "daily",
            "seconds": 86400,
            "description": "Daily bars for trend analysis",
            "typical_use": "Trend identification, position trading",
            "data_volume": "minimal",
            "polars_format": "1d",
        },
        "1w": {
            "category": "weekly",
            "seconds": 604800,
            "description": "Weekly bars for long-term structure",
            "typical_use": "Major trend analysis, portfolio allocation",
            "data_volume": "minimal",
            "polars_format": "1w",
        },
        "1m": {
            "category": "monthly",
            "seconds": 2592000,  # Approximate - varies by month
            "description": "Monthly bars for macro analysis",
            "typical_use": "Long-term investing, macro trends",
            "data_volume": "minimal",
            "polars_format": "1mo",
        },
        "1q": {
            "category": "quarterly",
            "seconds": 7776000,  # Approximate - 90 days
            "description": "Quarterly bars for fundamental analysis",
            "typical_use": "Business cycle analysis, long-term allocation",
            "data_volume": "minimal",
            "polars_format": "3mo",
        },
        "1y": {
            "category": "yearly",
            "seconds": 31536000,  # 365 days
            "description": "Yearly bars for multi-year analysis",
            "typical_use": "Multi-year trends, generational analysis",
            "data_volume": "minimal",
            "polars_format": "1y",
        },
    }

    @classmethod
    def validate_timeframe(cls, timeframe: str) -> bool:
        """Validate that the timeframe is supported (strict mode only)."""
        return timeframe in cls.TIMEFRAME_METADATA

    @classmethod
    def get_polars_format(cls, timeframe: str) -> str:
        """Get the Polars format for a timeframe."""
        metadata = cls.TIMEFRAME_METADATA.get(timeframe)
        if metadata:
            return metadata.get("polars_format", timeframe)
        return timeframe

    @classmethod
    def get_optimal_source_timeframe(cls, target_timeframe: str, available_timeframes: list[str]) -> str | None:
        """
        Get optimal source timeframe for aggregating to target.

        Args:
            target_timeframe: Target timeframe to aggregate to
            available_timeframes: List of available source timeframes

        Returns:
            Optimal source timeframe or None if target already exists or no valid source
        """
        # If target exists, use it directly (pass-through)
        if target_timeframe in available_timeframes:
            return target_timeframe

        target_metadata = cls.TIMEFRAME_METADATA.get(target_timeframe)
        if not target_metadata:
            return None

        target_seconds = target_metadata["seconds"]

        # Find all mathematically valid sources (those that divide evenly into target)
        valid_sources = []
        for source_tf in available_timeframes:
            source_metadata = cls.TIMEFRAME_METADATA.get(source_tf)
            if source_metadata:
                source_seconds = source_metadata["seconds"]
                if target_seconds % source_seconds == 0:
                    valid_sources.append((source_tf, source_seconds))

        if not valid_sources:
            return None

        # Return the source with the largest duration (minimize aggregation operations)
        return max(valid_sources, key=lambda x: x[1])[0]


class SwingPointsConfig(BaseModel):
    """Configuration for swing point detection with comprehensive parameter documentation."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")

    window: int = Field(
        default=5,
        ge=3,  # Replaces manual validator
        description="Number of bars for swing detection window (must be odd for symmetry)",
        examples=[3, 5, 7, 9],
        json_schema_extra={
            "recommendation": "Use odd numbers for symmetric lookback/forward windows",
            "impact": "Larger windows = fewer but more significant swings",
        },
    )
    threshold: float = Field(
        default=5.0,
        ge=0,  # Replaces manual validator
        description="Minimum percentage move to qualify as swing point",
        examples=[1.0, 2.5, 5.0, 10.0],
        json_schema_extra={
            "unit": "percent",
            "asset_class_recommendations": {
                "crypto": "3-5% (high volatility)",
                "equities": "1-3% (medium volatility)",
                "fx": "0.5-1.5% (lower volatility)",
            },
        },
    )


class GapDetectionConfig(BaseModel):
    """Configuration for gap detection with comprehensive threshold documentation."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")

    threshold: float = Field(
        default=0.0,
        ge=0,  # Replaces manual validator
        description="Minimum gap size as decimal percentage (0.01 = 1% gap, 0 = any gap)",
        examples=[0.0, 0.001, 0.005, 0.01, 0.02],
        json_schema_extra={
            "unit": "decimal_percentage",
            "conversion": "0.01 = 1%, 0.001 = 0.1%, 0 = any gap",
            "asset_class_recommendations": {
                "crypto": "0.005-0.02 (larger gaps expected)",
                "equities": "0.001-0.005 (smaller gaps typical)",
                "fx": "0.0001-0.001 (very small gaps)",
                "any_gap": "0 (detects all gaps regardless of size)",
            },
        },
    )


class TimeframeItemConfig(BaseModel):
    """Configuration for a single timeframe item with flexible timeframe targeting."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")

    timeframes: list[str] = Field(
        description="List of timeframes to apply this configuration to",
        min_length=1,
        examples=[["5m", "15m"], ["1h", "4h"], ["all"]],
        json_schema_extra={
            "special_values": {"all": "Apply to all available timeframes"},
            "validation": "Cannot mix 'all' with specific timeframes",
            "supported_timeframes": list(TimeframeConfig.TIMEFRAME_METADATA.keys()),
        },
    )
    swing_points: SwingPointsConfig | None = Field(
        default=None, description="Optional swing point detection configuration for these timeframes"
    )
    gap_detection: GapDetectionConfig | None = Field(
        default=None,
        description="Optional gap detection configuration for these timeframes",
        json_schema_extra={"note": "Most useful for session-based markets (equities) with overnight gaps"},
    )

    @field_validator("timeframes")
    @classmethod
    def validate_timeframes(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("timeframes cannot be empty")

        # Validate each timeframe string
        for tf in v:
            if not isinstance(tf, str) or not tf.strip():
                raise ValueError("all timeframes must be non-empty strings")

        return v

    @model_validator(mode="after")
    def validate_timeframe_combinations(self) -> Self:
        """Validate that 'all' is not mixed with specific timeframes."""
        if "all" in self.timeframes and len(self.timeframes) > 1:
            raise ValueError("'all' cannot be combined with specific timeframes")
        return self


class IndicatorsConfig(BaseModel):
    """Complete configuration for indicators component with per-timeframe settings."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")

    timeframe_configs: list[TimeframeItemConfig] = Field(
        description="List of timeframe-specific indicator configurations",
        min_length=1,
        json_schema_extra={
            "pattern": "Each config applies to specific timeframes",
            "flexibility": "Different parameters for different timeframe groups",
            "minimum": "At least one timeframe config required",
        },
    )


class AggregationConfig(BaseModel):
    """Complete configuration for aggregation component with market-aware settings."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")

    target_timeframes: list[str] = Field(
        description="List of target timeframes for OHLC aggregation",
        min_length=1,
        examples=[["5m"], ["1h", "4h"], ["5m", "15m", "1h", "1d"]],
        json_schema_extra={
            "supported_formats": list(TimeframeConfig.TIMEFRAME_METADATA.keys()),
            "multi_timeframe": "Process multiple timeframes in single operation",
            "validation": "All timeframes must be valid and supported",
        },
    )
    asset_class: Literal["crypto", "equities", "fx"] = Field(
        default="equities",
        description="Asset class determining market behavior and defaults",
        json_schema_extra={
            "impact": "Determines timezone handling, session alignment, and gap processing",
            "registry_reference": "See AssetClassConfig.REGISTRY for detailed characteristics",
        },
    )
    timezone: str | None = Field(
        default=None,  # Explicit default for static type checkers (Pylance), model validator applies asset class defaults
        description="Override timezone (None uses asset class default)",
        examples=["US/Eastern", "Europe/London", "Asia/Tokyo", "UTC"],
        json_schema_extra={
            "override_behavior": "When None, uses asset class default timezone",
            "validation": "Must be valid pytz timezone if specified",
            "special_cases": {"crypto": "Always UTC regardless of override", "fx": "Always UTC regardless of override"},
        },
    )
    hour_boundary: bool | None = Field(
        default=None,  # Explicit default for static type checkers (Pylance), model validator applies asset class defaults
        description="Override hour boundary alignment (None uses asset class default)",
        json_schema_extra={
            "true": "Align to hour boundaries (00:00, 01:00, 02:00...)",
            "false": "Align to session_start offset",
            "auto": "When None, determined by asset class characteristics",
        },
    )
    session_start: str | None = Field(
        default=None,  # Explicit default for static type checkers (Pylance), model validator applies asset class defaults
        description="Override session start time HH:MM (None uses asset class default)",
        pattern=r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$",
        examples=["09:30", "14:30", "00:00"],
        json_schema_extra={
            "format": "24-hour HH:MM format",
            "override_behavior": "When None, uses asset class default session_start",
        },
    )

    @field_validator("target_timeframes")
    @classmethod
    def validate_and_expand_target_timeframes(cls, target_timeframes: list[str]) -> list[str]:
        """
        Validate target_timeframes field and expand 'all' keyword.

        This Pydantic field validator is automatically called when AggregationConfig
        is instantiated. It validates individual timeframes and expands ['all'] to
        all supported timeframes.
        """
        if not target_timeframes:
            raise ValueError("target_timeframes cannot be empty")

        # Check for 'all' keyword
        if "all" in target_timeframes:
            if len(target_timeframes) > 1:
                raise ValueError("'all' cannot be combined with specific timeframes")
            # Expand to all supported timeframes
            return list(TimeframeConfig.TIMEFRAME_METADATA.keys())

        # Validate each specific timeframe
        for i, timeframe in enumerate(target_timeframes):
            if not isinstance(timeframe, str) or not timeframe.strip():
                raise ValueError(f"target_timeframes[{i}] must be a non-empty string")

            # Validate timeframe format using TimeframeConfig
            if not TimeframeConfig.validate_timeframe(timeframe):
                raise ValueError(f"Invalid timeframe '{timeframe}'")

        return target_timeframes

    @model_validator(mode="before")
    @classmethod
    def apply_asset_class_defaults(cls, data: Any) -> Any:
        """Apply AssetClassConfig defaults for timezone, hour_boundary, and session_start."""
        if not isinstance(data, dict):
            return data

        # Get asset class, default to "equities"
        asset_class = data.get("asset_class", "equities")
        asset_config = AssetClassConfig.get_config(asset_class)

        # Force UTC timezone for crypto and fx regardless of input
        if asset_class in ["crypto", "fx"]:
            data["timezone"] = "UTC"
        elif "timezone" not in data or data["timezone"] is None:
            data["timezone"] = asset_config.timezone

        # Apply other defaults only if fields are not provided or are None
        if "hour_boundary" not in data or data["hour_boundary"] is None:
            data["hour_boundary"] = asset_config.hour_boundary
        if "session_start" not in data or data["session_start"] is None:
            data["session_start"] = asset_config.session_start

        return data

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str | None) -> str | None:
        if v is not None:
            if not isinstance(v, str):
                raise ValueError("timezone must be a string or None")

            # Validate timezone using pytz
            try:
                pytz.timezone(v)
            except pytz.UnknownTimeZoneError as err:
                raise ValueError(f"Unknown timezone: {v}") from err

        return v

    @field_validator("session_start")
    @classmethod
    def validate_session_start(cls, v: str | None) -> str | None:
        if v is not None:
            if not isinstance(v, str):
                raise ValueError("session_start must be a string or None")

            # Basic time format validation (HH:MM)
            time_pattern = r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$"
            if not re.match(time_pattern, v):
                raise ValueError("session_start must be in HH:MM format")

        return v


class FactoryConfig(BaseModel):
    """Root configuration for Factory.create_all method with complete pipeline setup."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")

    aggregation: AggregationConfig = Field(
        description="Aggregation component configuration for OHLC timeframe conversion",
        json_schema_extra={
            "purpose": "Converts input data to target timeframes with market-aware alignment",
            "output": "Multi-timeframe OHLC data with timeframe column",
        },
    )
    indicators: IndicatorsConfig = Field(
        description="Indicators component configuration for TheStrat analysis",
        json_schema_extra={
            "purpose": "Applies TheStrat indicators to aggregated data",
            "output": "Enhanced data with swing points, patterns, and signals",
        },
    )

    @model_validator(mode="after")
    def validate_configuration_consistency(self) -> Self:
        """Validate consistency between aggregation and indicators configurations."""
        # Future enhancement: Could validate that indicator timeframes are compatible
        # with aggregation target timeframes, but this is not currently enforced
        # in the existing system
        return self


# Convenience type aliases for easier importing
SwingPoints = SwingPointsConfig
GapDetection = GapDetectionConfig
TimeframeItem = TimeframeItemConfig
Indicators = IndicatorsConfig
Aggregation = AggregationConfig
Factory = FactoryConfig


# =============================================================================
# DataFrame Schema Models
# =============================================================================


class IndicatorSchema(BaseModel):
    """
    **Complete Indicator Schema**

    Defines all columns that are created by TheStrat processing pipeline.
    All columns are required as the indicators component creates them all.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")

    # Base OHLCV+Symbol columns (input data)
    timestamp: datetime = Field(
        description="Timestamp for each bar/candle",
        json_schema_extra={"polars_dtype": Datetime, "input": True, "category": "base_ohlc", "nullable": False},
    )
    open: float = Field(
        description="Opening price for the time period",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "input": True, "category": "base_ohlc", "nullable": False},
    )
    high: float = Field(
        description="Highest price during the time period",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "input": True, "category": "base_ohlc", "nullable": False},
    )
    low: float = Field(
        description="Lowest price during the time period",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "input": True, "category": "base_ohlc", "nullable": False},
    )
    close: float = Field(
        description="Closing price for the time period",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "input": True, "category": "base_ohlc", "nullable": False},
    )
    symbol: str | None = Field(
        default=None,
        description="Trading symbol or ticker (e.g., 'AAPL', 'BTC-USD')",
        json_schema_extra={
            "polars_dtype": String,
            "input": True,
            "category": "base_ohlc",
            "optional": True,
            "nullable": True,
        },
    )
    volume: float | None = Field(
        default=None,
        description="Trading volume for the time period (supports fractional shares/units)",
        ge=0,
        json_schema_extra={
            "polars_dtype": Float64,
            "input": True,
            "category": "base_ohlc",
            "optional": True,
            "nullable": True,
        },
    )
    timeframe: str = Field(
        description="Timeframe identifier (e.g., '5min', '1h', '1d')",
        json_schema_extra={
            "polars_dtype": String,
            "input": True,
            "category": "base_ohlc",
            "note": "Added by Aggregation component for multi-timeframe data",
            "nullable": False,
        },
    )

    # Price Analysis Columns (output)
    percent_close_from_high: float = Field(
        description="Percentage of close price from bar high (0-100%)",
        ge=0,
        le=100,
        json_schema_extra={
            "polars_dtype": Float64,
            "output": True,
            "category": "price_analysis",
            "calculation": "((high - close) / (high - low)) * 100",
            "nullable": False,
        },
    )
    percent_close_from_low: float = Field(
        description="Percentage of close price from bar low (0-100%)",
        ge=0,
        le=100,
        json_schema_extra={
            "polars_dtype": Float64,
            "output": True,
            "category": "price_analysis",
            "calculation": "((close - low) / (high - low)) * 100",
            "nullable": False,
        },
    )
    ath: float = Field(
        description="All-time high (cumulative maximum of high prices)",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "output": True, "category": "price_analysis", "nullable": False},
    )
    atl: float = Field(
        description="All-time low (cumulative minimum of low prices)",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "output": True, "category": "price_analysis", "nullable": False},
    )
    new_ath: bool = Field(
        description="True when current high equals all-time high",
        json_schema_extra={"polars_dtype": Boolean, "output": True, "category": "price_analysis", "nullable": False},
    )
    new_atl: bool = Field(
        description="True when current low equals all-time low",
        json_schema_extra={"polars_dtype": Boolean, "output": True, "category": "price_analysis", "nullable": False},
    )

    # Gap Detection Columns
    gapper: int = Field(
        description="Threshold-based gap detection: 1=gap up above threshold, 0=gap down below threshold, null=no significant gap",
        json_schema_extra={"polars_dtype": Int32, "output": True, "category": "gap_detection", "nullable": True},
    )

    # Market Structure Analysis Columns
    higher_high: float = Field(
        description="Higher high price in uptrend structure",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "output": True, "category": "market_structure", "nullable": True},
    )
    lower_high: float = Field(
        description="Lower high price in downtrend structure",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "output": True, "category": "market_structure", "nullable": True},
    )
    higher_low: float = Field(
        description="Higher low price in uptrend structure",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "output": True, "category": "market_structure", "nullable": True},
    )
    lower_low: float = Field(
        description="Lower low price in downtrend structure",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "output": True, "category": "market_structure", "nullable": True},
    )

    # TheStrat Pattern Columns
    continuity: int = Field(
        description="Single bar open/close relationship",
        ge=-1,
        le=1,
        json_schema_extra={
            "polars_dtype": Int32,
            "output": True,
            "category": "thestrat_patterns",
            "nullable": False,
            "values": {
                1: "Close > Open (bullish candle)",
                0: "Close < Open (bearish candle)",
                -1: "Close = Open (doji candle)",
            },
        },
    )
    in_force: bool = Field(
        description="True when price action confirms directional bias",
        json_schema_extra={"polars_dtype": Boolean, "output": True, "category": "thestrat_patterns", "nullable": False},
    )
    scenario: str = Field(
        description="TheStrat scenario classification (1, 2U, 2D, 3)",
        json_schema_extra={
            "polars_dtype": String,
            "output": True,
            "category": "thestrat_patterns",
            "nullable": True,
            "values": ["1", "2U", "2D", "3"],
        },
    )

    # Signal Columns
    signal: str = Field(
        description="Detected TheStrat signal pattern",
        json_schema_extra={"polars_dtype": String, "output": True, "category": "signals", "nullable": True},
    )
    type: str = Field(
        description="Signal type/category (e.g., 'reversal', 'continuation')",
        json_schema_extra={"polars_dtype": String, "output": True, "category": "signals", "nullable": True},
    )
    bias: str = Field(
        description="Signal directional bias ('long' or 'short')",
        json_schema_extra={
            "polars_dtype": String,
            "output": True,
            "category": "signals",
            "nullable": True,
            "values": ["long", "short"],
        },
    )
    signal_json: str = Field(
        description="Complete signal metadata as JSON string",
        json_schema_extra={"polars_dtype": String, "output": True, "category": "signals", "nullable": True},
    )

    # Special Pattern Columns
    hammer: bool = Field(
        description="True when hammer/doji pattern detected",
        json_schema_extra={"polars_dtype": Boolean, "output": True, "category": "special_patterns", "nullable": False},
    )
    shooter: bool = Field(
        description="True when shooting star pattern detected",
        json_schema_extra={"polars_dtype": Boolean, "output": True, "category": "special_patterns", "nullable": False},
    )
    kicker: int = Field(
        description="Kicker pattern detection (1=bullish kicker, 0=bearish kicker, null=no kicker)",
        json_schema_extra={"polars_dtype": Int32, "output": True, "category": "special_patterns", "nullable": True},
    )
    f23: bool = Field(
        description="True when F23 pattern detected",
        json_schema_extra={"polars_dtype": Boolean, "output": True, "category": "special_patterns", "nullable": False},
    )
    f23x: str = Field(
        description="F23 pattern variant (F23U, F23D)",
        json_schema_extra={
            "polars_dtype": String,
            "output": True,
            "category": "special_patterns",
            "nullable": True,
            "values": ["F23U", "F23D"],
        },
    )
    f23_trigger: float = Field(
        description="F23 trigger price level",
        gt=0,
        json_schema_extra={"polars_dtype": Float64, "output": True, "category": "special_patterns", "nullable": True},
    )
    pmg: int = Field(
        description="Price Magnitude Gap pattern result - cumulative directional count",
        json_schema_extra={"polars_dtype": Int32, "output": True, "category": "special_patterns", "nullable": False},
    )

    # Mother Bar Analysis Columns
    motherbar_problems: bool = Field(
        description="True when mother bar analysis has issues/conflicts",
        json_schema_extra={"polars_dtype": Boolean, "output": True, "category": "mother_bar", "nullable": False},
    )

    @classmethod
    def get_column_descriptions(cls) -> dict[str, str]:
        """
        Get descriptions for all possible DataFrame columns.

        Returns:
            Dictionary mapping column names to their descriptions
        """
        descriptions = {}

        # Get descriptions from the schema
        for field_name, field_info in cls.model_fields.items():
            if field_info.description:
                descriptions[field_name] = field_info.description

        return descriptions

    @classmethod
    def get_polars_dtypes(cls) -> dict[str, Any]:
        """
        Get Polars data types for all DataFrame columns.

        Returns:
            Dictionary mapping column names to their Polars data types
        """
        types = {}

        for field_name, field_info in cls.model_fields.items():
            json_extra = getattr(field_info, "json_schema_extra", {})
            if isinstance(json_extra, dict) and "polars_dtype" in json_extra:
                types[field_name] = json_extra["polars_dtype"]

        return types

    @classmethod
    def validate_dataframe(cls, df) -> dict[str, Any]:
        """
        Validate input DataFrame columns and data types against IndicatorSchema input requirements.

        Automatically converts Pandas DataFrames to Polars for consistent validation.

        Args:
            df: Polars or Pandas DataFrame to validate

        Returns:
            Dictionary with validation results including missing/extra columns, type issues,
            and the converted Polars DataFrame if conversion occurred
        """
        from polars import from_pandas

        # Detect DataFrame type and convert if necessary
        df_type = "unknown"
        converted_df = None
        conversion_errors = []

        if hasattr(df, "columns"):
            # Check if it's a Pandas DataFrame
            if hasattr(df, "dtypes") and not hasattr(df, "schema"):
                df_type = "pandas"
                try:
                    # Convert Pandas to Polars
                    converted_df = from_pandas(df)
                    df = converted_df  # Use converted DataFrame for validation
                except Exception as e:
                    conversion_errors.append(f"Failed to convert Pandas to Polars: {str(e)}")
                    # Fall back to column-only validation
                    df_columns = list(df.columns)
                    return {
                        "valid": False,
                        "conversion_error": conversion_errors[0],
                        "df_type": df_type,
                        "columns": df_columns,
                        "message": "Could not convert Pandas DataFrame to Polars for full validation",
                    }
            # Check if it's already a Polars DataFrame
            elif hasattr(df, "schema"):
                df_type = "polars"
            else:
                raise ValueError("Unknown DataFrame type - must be Pandas or Polars DataFrame")

            df_columns = list(df.columns)
        else:
            raise ValueError("Input must be a DataFrame with .columns attribute")

        required_fields = []
        optional_fields = []
        expected_types = {}

        # Extract input field requirements from schema
        for field_name, field_info in cls.model_fields.items():
            json_extra = getattr(field_info, "json_schema_extra", {})
            if isinstance(json_extra, dict) and json_extra.get("input"):
                if field_info.is_required():
                    required_fields.append(field_name)
                else:
                    optional_fields.append(field_name)

                # Store expected Polars type for validation
                if "polars_dtype" in json_extra:
                    expected_types[field_name] = json_extra["polars_dtype"]

        # Check for missing and extra columns
        missing_required = [col for col in required_fields if col not in df_columns]
        missing_optional = [col for col in optional_fields if col not in df_columns]
        extra_columns = [col for col in df_columns if col not in required_fields + optional_fields]

        # Check data types for present columns (now guaranteed to be Polars)
        type_issues = []
        if hasattr(df, "schema"):  # Polars DataFrame
            for col_name, expected_type in expected_types.items():
                if col_name in df_columns:
                    actual_type = df.schema[col_name]
                    if actual_type != expected_type:
                        type_issues.append(
                            {
                                "column": col_name,
                                "expected": expected_type.__name__
                                if hasattr(expected_type, "__name__")
                                else str(expected_type),
                                "actual": str(actual_type),
                            }
                        )

        result = {
            "valid": len(missing_required) == 0 and len(type_issues) == 0,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "extra_columns": extra_columns,
            "type_issues": type_issues,
            "required_fields": required_fields,
            "optional_fields": optional_fields,
            "expected_types": {k: v.__name__ if hasattr(v, "__name__") else str(v) for k, v in expected_types.items()},
            "df_type": df_type,
        }

        # Include converted DataFrame if conversion occurred
        if converted_df is not None:
            result["converted_df"] = converted_df
            result["conversion_performed"] = True
        else:
            result["conversion_performed"] = False

        return result

    @classmethod
    def get_column_categories(cls) -> dict[str, list[str]]:
        """
        Get columns organized by functional categories.

        Dynamically extracts categories from the IndicatorSchema metadata.

        Returns:
            Dictionary mapping category names to lists of column names
        """
        categories: dict[str, list[str]] = {}

        for field_name, field_info in cls.model_fields.items():
            json_extra = getattr(field_info, "json_schema_extra", {})
            if isinstance(json_extra, dict) and "category" in json_extra:
                category = json_extra["category"]
                if category not in categories:
                    categories[category] = []
                categories[category].append(field_name)

        # Sort columns within each category for consistent output
        for category in categories:
            categories[category].sort()

        return categories

    @classmethod
    def get_required_input_columns(cls) -> list[str]:
        """
        Get list of required input columns based on schema definition.

        Returns:
            List of column names that are required for input data
        """
        from pydantic_core import PydanticUndefined

        required_columns = []
        for field_name, field_info in cls.model_fields.items():
            json_extra = getattr(field_info, "json_schema_extra", {}) or {}
            if (
                json_extra.get("input") is True  # Is an input column
                and not json_extra.get("optional", False)  # Not marked as optional
                and getattr(field_info, "default", PydanticUndefined)
                is PydanticUndefined  # No default value (required)
            ):
                required_columns.append(field_name)
        return sorted(required_columns)

    @classmethod
    def get_optional_input_columns(cls) -> list[str]:
        """
        Get list of optional input columns based on schema definition.

        Returns:
            List of column names that are optional for input data
        """
        from pydantic_core import PydanticUndefined

        optional_columns = []
        for field_name, field_info in cls.model_fields.items():
            json_extra = getattr(field_info, "json_schema_extra", {}) or {}
            if (
                json_extra.get("input") is True  # Is an input column
                and (
                    json_extra.get("optional", False)  # Marked as optional
                    or getattr(field_info, "default", PydanticUndefined) is not PydanticUndefined
                )  # Has default value
            ):
                optional_columns.append(field_name)
        return sorted(optional_columns)

    @classmethod
    def get_all_input_columns(cls) -> list[str]:
        """
        Get list of all input columns (required + optional).

        Returns:
            List of all input column names
        """
        return sorted(cls.get_required_input_columns() + cls.get_optional_input_columns())

    @classmethod
    def get_field_metadata(cls, field_name: str) -> dict[str, Any]:
        """
        Get json_schema_extra metadata for a field, safely handling missing data.

        Args:
            field_name: Name of the field to get metadata for

        Returns:
            Dictionary of metadata from json_schema_extra, empty dict if not found
        """
        field_info = cls.model_fields.get(field_name)
        if not field_info:
            return {}
        return getattr(field_info, "json_schema_extra", {}) or {}
