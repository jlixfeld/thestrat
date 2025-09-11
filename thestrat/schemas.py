"""
Pydantic schema models for TheStrat configuration validation.

This module provides comprehensive validation schemas that replace all manual validation
logic in the Factory class. Models use Pydantic v2 features for maximum performance,
type safety, and detailed error reporting.
"""

import re
from typing import Any, ClassVar, Literal, get_args, get_origin

import pytz
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from typing_extensions import Self


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

    # Registry pattern - similar to TimeframeConfig.TIMEFRAME_TO_POLARS
    REGISTRY: ClassVar[dict[str, "AssetClassConfig"]] = {}


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

    # Timeframe mappings for aggregation
    TIMEFRAME_TO_POLARS: ClassVar[dict[str, str]] = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1h": "1h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "1w": "1w",
        "1m": "1mo",  # 1 month
        "1q": "3mo",  # 1 quarter = 3 months
        "1y": "1y",
    }

    # Comprehensive timeframe metadata
    TIMEFRAME_METADATA: ClassVar[dict[str, dict[str, Any]]] = {
        "1min": {
            "category": "sub-hourly",
            "seconds": 60,
            "description": "1-minute bars for high-frequency analysis",
            "typical_use": "Scalping, entry timing, tick analysis",
            "data_volume": "very_high",
        },
        "5min": {
            "category": "sub-hourly",
            "seconds": 300,
            "description": "5-minute bars for short-term patterns",
            "typical_use": "Day trading, quick reversals",
            "data_volume": "high",
        },
        "15min": {
            "category": "sub-hourly",
            "seconds": 900,
            "description": "15-minute bars for intraday structure",
            "typical_use": "Swing entries, session analysis",
            "data_volume": "medium",
        },
        "30min": {
            "category": "sub-hourly",
            "seconds": 1800,
            "description": "30-minute bars for session structure",
            "typical_use": "Half-hourly patterns, session transitions",
            "data_volume": "medium",
        },
        "1h": {
            "category": "hourly",
            "seconds": 3600,
            "description": "1-hour bars for daily structure analysis",
            "typical_use": "Day trading context, hourly levels",
            "data_volume": "low",
        },
        "4h": {
            "category": "multi-hourly",
            "seconds": 14400,
            "description": "4-hour bars for swing trading",
            "typical_use": "Swing trading, multi-day holds",
            "data_volume": "very_low",
        },
        "6h": {
            "category": "multi-hourly",
            "seconds": 21600,
            "description": "6-hour bars for extended swing analysis",
            "typical_use": "Position trading context",
            "data_volume": "very_low",
        },
        "12h": {
            "category": "multi-hourly",
            "seconds": 43200,
            "description": "12-hour bars for daily session analysis",
            "typical_use": "Asian/Western session splits",
            "data_volume": "very_low",
        },
        "1d": {
            "category": "daily",
            "seconds": 86400,
            "description": "Daily bars for trend analysis",
            "typical_use": "Trend identification, position trading",
            "data_volume": "minimal",
        },
        "1w": {
            "category": "weekly",
            "seconds": 604800,
            "description": "Weekly bars for long-term structure",
            "typical_use": "Major trend analysis, portfolio allocation",
            "data_volume": "minimal",
        },
        "1m": {
            "category": "monthly",
            "seconds": 2592000,  # Approximate - varies by month
            "description": "Monthly bars for macro analysis",
            "typical_use": "Long-term investing, macro trends",
            "data_volume": "minimal",
        },
        "1q": {
            "category": "quarterly",
            "seconds": 7776000,  # Approximate - 90 days
            "description": "Quarterly bars for fundamental analysis",
            "typical_use": "Business cycle analysis, long-term allocation",
            "data_volume": "minimal",
        },
        "1y": {
            "category": "yearly",
            "seconds": 31536000,  # 365 days
            "description": "Yearly bars for multi-year analysis",
            "typical_use": "Multi-year trends, generational analysis",
            "data_volume": "minimal",
        },
    }

    @classmethod
    def validate_timeframe(cls, timeframe: str) -> bool:
        """Validate that the timeframe is supported."""
        if timeframe in cls.TIMEFRAME_TO_POLARS:
            return True

        # For backward compatibility, also check polars-style patterns
        pattern = r"^(\d+)([a-zA-Z]+)$"
        match = re.match(pattern, timeframe)

        if not match:
            return False

        # Check if it matches a polars-style format that could be valid
        unit = match.group(2).lower()
        valid_polars_units = ["m", "h", "d", "w", "mo", "y"]
        return unit in valid_polars_units

    @classmethod
    def get_polars_format(cls, timeframe: str) -> str:
        """Get the Polars format for a timeframe."""
        return cls.TIMEFRAME_TO_POLARS.get(timeframe, timeframe)


def get_asset_class_config(asset_class: str) -> AssetClassConfig:
    """Get configuration for specific asset class."""
    return ASSET_CLASS_CONFIGS.get(asset_class, ASSET_CLASS_CONFIGS["equities"])


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
        default=0.001,
        ge=0,  # Replaces manual validator
        description="Minimum gap size as decimal percentage (0.01 = 1% gap)",
        examples=[0.001, 0.005, 0.01, 0.02],
        json_schema_extra={
            "unit": "decimal_percentage",
            "conversion": "0.01 = 1%, 0.001 = 0.1%",
            "asset_class_recommendations": {
                "crypto": "0.005-0.02 (larger gaps expected)",
                "equities": "0.001-0.005 (smaller gaps typical)",
                "fx": "0.0001-0.001 (very small gaps)",
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
            "supported_timeframes": list(TimeframeConfig.TIMEFRAME_TO_POLARS.keys()),
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
        examples=[
            [{"timeframes": ["all"], "swing_points": {"window": 5, "threshold": 2.0}}],
            [
                {"timeframes": ["5m"], "swing_points": {"window": 3, "threshold": 1.5}},
                {"timeframes": ["1h", "4h"], "swing_points": {"window": 7, "threshold": 3.0}},
            ],
        ],
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
            "supported_formats": list(TimeframeConfig.TIMEFRAME_TO_POLARS.keys()),
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
        default=None,
        description="Override timezone (None uses asset class default)",
        examples=["US/Eastern", "Europe/London", "Asia/Tokyo", "UTC"],
        json_schema_extra={
            "override_behavior": "When None, uses asset class default timezone",
            "validation": "Must be valid pytz timezone if specified",
            "special_cases": {"crypto": "Always UTC regardless of override", "fx": "Always UTC regardless of override"},
        },
    )
    hour_boundary: bool | None = Field(
        default=None,
        description="Override hour boundary alignment (None uses asset class default)",
        json_schema_extra={
            "true": "Align to hour boundaries (00:00, 01:00, 02:00...)",
            "false": "Align to session_start offset",
            "auto": "When None, determined by asset class characteristics",
        },
    )
    session_start: str | None = Field(
        default=None,
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
    def validate_target_timeframes(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("target_timeframes cannot be empty")

        for i, tf in enumerate(v):
            if not isinstance(tf, str) or not tf.strip():
                raise ValueError(f"target_timeframes[{i}] must be a non-empty string")

            # Validate timeframe format using TimeframeConfig
            if not TimeframeConfig.validate_timeframe(tf):
                raise ValueError(f"Invalid timeframe '{tf}'")

        return v

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


# Documentation generation utilities


class SchemaDocGenerator:
    """Generate comprehensive documentation from Pydantic schemas."""

    @staticmethod
    def generate_field_docs(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
        """
        Extract comprehensive field documentation from a Pydantic model.

        Args:
            model: Pydantic model class to document

        Returns:
            Dictionary mapping field names to their documentation metadata
        """
        field_docs = {}

        for field_name, field_info in model.model_fields.items():
            field_doc = {
                "name": field_name,
                "type": SchemaDocGenerator._get_type_string(field_info.annotation),
                "description": field_info.description or "No description provided",
                "default": SchemaDocGenerator._format_default(field_info.default),
                "required": field_info.is_required(),
                "examples": getattr(field_info, "examples", []),
            }

            # Add validation constraints
            constraints = SchemaDocGenerator._extract_constraints(field_info)
            if constraints:
                field_doc["constraints"] = constraints

            # Add json_schema_extra information
            if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                field_doc["metadata"] = field_info.json_schema_extra

            field_docs[field_name] = field_doc

        return field_docs

    @staticmethod
    def generate_markdown_table(model: type[BaseModel]) -> str:
        """
        Generate a markdown table for a Pydantic model's fields.

        Args:
            model: Pydantic model class to document

        Returns:
            Markdown table string
        """
        field_docs = SchemaDocGenerator.generate_field_docs(model)

        # Table header
        markdown = f"## {model.__name__}\n\n"
        markdown += f"{model.__doc__ or 'Configuration model'}\n\n"
        markdown += "| Field | Type | Default | Description |\n"
        markdown += "|-------|------|---------|-------------|\n"

        # Table rows
        for field_name, field_doc in field_docs.items():
            type_str = field_doc["type"]
            default_str = field_doc["default"]
            desc_str = field_doc["description"]

            # Add constraints to description if present
            if "constraints" in field_doc:
                constraints = field_doc["constraints"]
                constraint_parts = []
                if "min_length" in constraints:
                    constraint_parts.append(f"min length: {constraints['min_length']}")
                if "max_length" in constraints:
                    constraint_parts.append(f"max length: {constraints['max_length']}")
                if "ge" in constraints:
                    constraint_parts.append(f">= {constraints['ge']}")
                if "le" in constraints:
                    constraint_parts.append(f"<= {constraints['le']}")
                if "pattern" in constraints:
                    constraint_parts.append(f"pattern: `{constraints['pattern']}`")

                if constraint_parts:
                    desc_str += f" ({', '.join(constraint_parts)})"

            markdown += f"| `{field_name}` | {type_str} | {default_str} | {desc_str} |\n"

        # Add ClassVar information if present
        classvars = SchemaDocGenerator._get_classvars(model)
        if classvars:
            markdown += "\n### Class Constants\n\n"
            for var_name, var_info in classvars.items():
                markdown += f"- **`{var_name}`**: {var_info['description']}\n"
                if var_info.get("keys"):
                    markdown += f"  - Available keys: {', '.join(var_info['keys'])}\n"

                # Add detailed configuration tables for registries
                if var_name == "REGISTRY" and var_info.get("detailed_configs"):
                    markdown += "\n#### Configured Asset Classes\n\n"
                    for config_name, config_details in var_info["detailed_configs"].items():
                        markdown += f"**{config_name}**\n"
                        markdown += f"- Trading Hours: {config_details.get('trading_hours', 'N/A')}\n"
                        markdown += f"- Timezone: `{config_details.get('timezone', 'N/A')}`\n"
                        markdown += f"- Session Start: `{config_details.get('session_start', 'N/A')}`\n"
                        markdown += f"- Hour Boundary: `{config_details.get('hour_boundary', 'N/A')}`\n"
                        markdown += "\n"

                # Add detailed timeframe metadata tables
                elif var_name == "TIMEFRAME_METADATA" and var_info.get("detailed_metadata"):
                    markdown += "\n#### Timeframe Details\n\n"
                    markdown += "| Timeframe | Category | Duration | Description | Typical Use |\n"
                    markdown += "|-----------|----------|----------|-------------|-------------|\n"

                    for tf_name, tf_details in var_info["detailed_metadata"].items():
                        category = tf_details.get("category", "N/A")
                        seconds = tf_details.get("seconds", 0)
                        description = tf_details.get("description", "N/A")
                        typical_use = tf_details.get("typical_use", "N/A")

                        # Convert seconds to readable duration
                        if seconds < 3600:
                            duration = f"{seconds // 60}m"
                        elif seconds < 86400:
                            duration = f"{seconds // 3600}h"
                        elif seconds < 604800:
                            duration = f"{seconds // 86400}d"
                        else:
                            duration = f"{seconds // 604800}w"

                        markdown += f"| `{tf_name}` | {category} | {duration} | {description} | {typical_use} |\n"
                    markdown += "\n"

        return markdown

    @staticmethod
    def generate_json_schema(model: type[BaseModel]) -> dict[str, Any]:
        """
        Generate JSON Schema for a Pydantic model with all metadata.

        Args:
            model: Pydantic model class

        Returns:
            JSON Schema dictionary with enhanced metadata
        """
        schema = model.model_json_schema()

        # Enhance with field documentation
        field_docs = SchemaDocGenerator.generate_field_docs(model)

        if "properties" in schema:
            for field_name, field_schema in schema["properties"].items():
                if field_name in field_docs:
                    field_doc = field_docs[field_name]

                    # Add examples if available
                    if field_doc["examples"]:
                        field_schema["examples"] = field_doc["examples"]

                    # Add metadata
                    if "metadata" in field_doc:
                        field_schema.update(field_doc["metadata"])

        return schema

    @staticmethod
    def generate_all_model_docs() -> dict[str, dict[str, Any]]:
        """
        Generate documentation for all schema models.

        Returns:
            Dictionary mapping model names to their complete documentation
        """
        models = [
            AssetClassConfig,
            TimeframeConfig,
            SwingPointsConfig,
            GapDetectionConfig,
            TimeframeItemConfig,
            IndicatorsConfig,
            AggregationConfig,
            FactoryConfig,
        ]

        all_docs = {}
        for model in models:
            all_docs[model.__name__] = {
                "class_doc": model.__doc__,
                "field_docs": SchemaDocGenerator.generate_field_docs(model),
                "markdown": SchemaDocGenerator.generate_markdown_table(model),
                "json_schema": SchemaDocGenerator.generate_json_schema(model),
            }

        return all_docs

    @staticmethod
    def _get_type_string(annotation: Any) -> str:
        """Convert type annotation to readable string."""
        if annotation is None:
            return "None"

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            return getattr(annotation, "__name__", str(annotation))

        if origin is list:
            if args:
                return f"list[{SchemaDocGenerator._get_type_string(args[0])}]"
            return "list"

        if origin is dict:
            if len(args) == 2:
                return f"dict[{SchemaDocGenerator._get_type_string(args[0])}, {SchemaDocGenerator._get_type_string(args[1])}]"
            return "dict"

        if origin is type(None) or str(origin) == "UnionType":  # Union/Optional
            if len(args) == 2 and type(None) in args:
                # This is Optional[T]
                non_none = [arg for arg in args if arg is not type(None)][0]
                return f"{SchemaDocGenerator._get_type_string(non_none)} | None"
            else:
                # This is Union[T, U, ...]
                arg_strs = [SchemaDocGenerator._get_type_string(arg) for arg in args]
                return " | ".join(arg_strs)

        if hasattr(origin, "__name__"):
            if args:
                arg_strs = [SchemaDocGenerator._get_type_string(arg) for arg in args]
                return f"{origin.__name__}[{', '.join(arg_strs)}]"
            return origin.__name__

        return str(annotation)

    @staticmethod
    def _format_default(default: Any) -> str:
        """Format default value for documentation."""
        if default is ...:  # Ellipsis means required field
            return "*required*"
        if default is None:
            return "`None`"
        if isinstance(default, str):
            return f'`"{default}"`'
        if isinstance(default, (int, float, bool)):
            return f"`{default}`"
        if isinstance(default, (list, dict)):
            return f"`{default}`"

        return str(default)

    @staticmethod
    def _extract_constraints(field_info: FieldInfo) -> dict[str, Any]:
        """Extract validation constraints from field info."""
        constraints = {}

        # Common Pydantic constraints
        constraint_attrs = ["min_length", "max_length", "ge", "le", "gt", "lt", "pattern"]
        for attr in constraint_attrs:
            if hasattr(field_info, attr):
                value = getattr(field_info, attr)
                if value is not None:
                    constraints[attr] = value

        return constraints

    @staticmethod
    def _get_classvars(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
        """Extract ClassVar information from model."""
        classvars = {}

        # Check for known ClassVars using getattr to avoid type checker issues
        timeframe_to_polars = getattr(model, "TIMEFRAME_TO_POLARS", None)
        if timeframe_to_polars:
            classvars["TIMEFRAME_TO_POLARS"] = {
                "description": "Mapping of timeframe strings to Polars format strings",
                "keys": list(timeframe_to_polars.keys()),
            }

        timeframe_metadata = getattr(model, "TIMEFRAME_METADATA", None)
        if timeframe_metadata:
            classvars["TIMEFRAME_METADATA"] = {
                "description": "Comprehensive metadata for each supported timeframe",
                "keys": list(timeframe_metadata.keys()),
                "detailed_metadata": timeframe_metadata,
            }

        registry = getattr(model, "REGISTRY", None)
        if registry:
            # Extract detailed configuration information
            detailed_configs = {}
            for key, config_obj in registry.items():
                if hasattr(config_obj, "model_dump"):
                    detailed_configs[key] = config_obj.model_dump()
                else:
                    detailed_configs[key] = dict(config_obj) if isinstance(config_obj, dict) else {}

            classvars["REGISTRY"] = {
                "description": "Registry of predefined configurations for each asset class",
                "keys": list(registry.keys()),
                "detailed_configs": detailed_configs,
            }

        return classvars


def generate_complete_documentation() -> str:
    """
    Generate complete markdown documentation for all schema models.

    Returns:
        Complete markdown documentation string
    """
    docs = SchemaDocGenerator.generate_all_model_docs()

    markdown = "# TheStrat Schema Documentation\n\n"
    markdown += "Auto-generated documentation for all Pydantic configuration models.\n\n"
    markdown += "---\n\n"

    for model_docs in docs.values():
        markdown += model_docs["markdown"] + "\n\n"

    return markdown


def export_json_schemas() -> dict[str, dict[str, Any]]:
    """
    Export JSON schemas for all models (useful for OpenAPI/AsyncAPI generation).

    Returns:
        Dictionary mapping model names to their JSON schemas
    """
    docs = SchemaDocGenerator.generate_all_model_docs()
    return {name: doc_info["json_schema"] for name, doc_info in docs.items()}
