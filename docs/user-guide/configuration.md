# Configuration Guide

TheStrat uses Pydantic models for type-safe, validated configuration. All components are created through the Factory pattern with comprehensive validation.

## Configuration Overview

The configuration system uses a hierarchical structure:

```
FactoryConfig
├── AggregationConfig (timeframes, asset class, timezone)
└── IndicatorsConfig
    └── TimeframeItemConfig[] (per-timeframe settings)
        ├── SwingPointsConfig
        ├── GapDetectionConfig
        └── TargetConfig
```

## Factory Configuration

### FactoryConfig

The top-level configuration that creates a complete processing pipeline:

```python
from thestrat.schemas import FactoryConfig, AggregationConfig, IndicatorsConfig

config = FactoryConfig(
    aggregation=AggregationConfig(...),  # Required
    indicators=IndicatorsConfig(...)      # Required
)

pipeline = Factory.create_all(config)
```

**Fields:**
- `aggregation` (AggregationConfig): Configuration for timeframe aggregation
- `indicators` (IndicatorsConfig): Configuration for TheStrat indicators

## Aggregation Configuration

### AggregationConfig

Controls how OHLCV data is aggregated across timeframes:

```python
from thestrat.schemas import AggregationConfig

config = AggregationConfig(
    target_timeframes=["5min", "15min", "1h"],  # Required
    asset_class="equities",                      # Default: "equities"
    timezone="US/Eastern",                       # Default: from asset_class
    hour_boundary=False,                         # Default: from asset_class
    session_start="09:30"                        # Default: from asset_class
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_timeframes` | `list[str]` | Required | List of target timeframes (e.g., `["5min", "1h"]`) |
| `asset_class` | `str` | `"equities"` | Asset class: `"equities"`, `"crypto"`, or `"fx"` |
| `timezone` | `str` | From asset class | Timezone for timestamp handling (e.g., `"US/Eastern"`) |
| `hour_boundary` | `bool` | From asset class | Align hourly+ timeframes to hour boundaries |
| `session_start` | `str` | From asset class | Session start time in HH:MM format (e.g., `"09:30"`) |

**Supported Timeframes:**

```python
# Valid timeframe strings
["1min", "5min", "15min", "30min", "1h", "4h", "6h", "12h", 
 "1d", "1w", "1m", "1q", "1y"]
```

**Asset Class Defaults:**

=== "Equities"
    - **Timezone**: `US/Eastern`
    - **Session Start**: `09:30`
    - **Hour Boundary**: `False` (align to session)
    - **Trading Hours**: 9:30 AM - 4:00 PM ET

=== "Crypto"
    - **Timezone**: `UTC` (forced, cannot override)
    - **Session Start**: `00:00`
    - **Hour Boundary**: `True` (align to hour boundaries)
    - **Trading Hours**: 24/7 continuous

=== "Forex"
    - **Timezone**: `UTC` (forced, cannot override)
    - **Session Start**: `00:00`
    - **Hour Boundary**: `True` (align to hour boundaries)
    - **Trading Hours**: 24/5 (Sunday 5 PM - Friday 5 PM ET)

!!! note "Timezone Enforcement"
    Crypto and FX **always use UTC** regardless of timezone parameter. Only equities allow timezone customization.

### Timezone Handling Rules

**Input Data Timezone Conversion:**

| Input Timestamp State | Aggregation Behavior |
|----------------------|---------------------|
| Naive (no timezone) | Assumes already in target timezone, adds awareness with warning |
| Aware (different from target) | Converts to target timezone |
| Aware (same as target) | No conversion needed |

**Example:**

```python
import polars as pl

# Database data in UTC - convert before aggregation
df = pl.read_database("SELECT * FROM ohlcv", connection)
df = df.with_columns([
    pl.col("timestamp")
        .dt.replace_time_zone("UTC")          # Mark as UTC
        .dt.convert_time_zone("US/Eastern")   # Convert to Eastern
])

# Now feed to Factory
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5min", "1h"],
        asset_class="equities",
        timezone="US/Eastern"
    ),
    indicators=IndicatorsConfig(...)
)

pipeline = Factory.create_all(config)
aggregated = pipeline["aggregation"].process(df)
```

## Indicators Configuration

### IndicatorsConfig

Controls TheStrat indicator calculation with per-timeframe settings:

```python
from thestrat.schemas import (
    IndicatorsConfig, TimeframeItemConfig,
    SwingPointsConfig, GapDetectionConfig, TargetConfig
)

config = IndicatorsConfig(
    timeframe_configs=[
        TimeframeItemConfig(
            timeframes=["5min"],
            swing_points=SwingPointsConfig(window=3, threshold=1.5),
            gap_detection=GapDetectionConfig(enabled=True),
            target_config=TargetConfig(max_targets=3)
        ),
        TimeframeItemConfig(
            timeframes=["15min", "1h"],
            swing_points=SwingPointsConfig(window=7, threshold=2.5)
        ),
        TimeframeItemConfig(
            timeframes=["all"]  # Apply to all unspecified timeframes
        )
    ]
)
```

**Fields:**
- `timeframe_configs` (list[TimeframeItemConfig]): Per-timeframe configuration list

### TimeframeItemConfig

Configuration for specific timeframes or `"all"` as fallback:

```python
TimeframeItemConfig(
    timeframes=["5min", "15min"],  # Or ["all"] for catch-all
    swing_points=SwingPointsConfig(...),  # Optional
    gap_detection=GapDetectionConfig(...),  # Optional
    target_config=TargetConfig(...)  # Optional
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeframes` | `list[str]` | Required | List of timeframes or `["all"]` for unspecified |
| `swing_points` | `SwingPointsConfig` | Defaults | Swing point detection settings |
| `gap_detection` | `GapDetectionConfig` | Defaults | Gap detection settings |
| `target_config` | `TargetConfig` | Defaults | Target price calculation settings |

!!! tip "Using 'all' Timeframes"
    Use `timeframes=["all"]` as a catch-all for any timeframes not explicitly configured. If present, it must be the last config item.

### SwingPointsConfig

Controls peak/valley detection for market structure:

```python
SwingPointsConfig(
    window=5,       # Look 5 bars back and ahead
    threshold=2.0   # Require 2% price change to confirm
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `window` | `int` | `5` | Number of bars to look back/ahead for peak/valley detection |
| `threshold` | `float` | `2.0` | Minimum percentage price change to confirm swing point |

**Window Sizing Guide:**

| Trading Style | Recommended Window | Threshold |
|--------------|-------------------|-----------|
| Scalping | 3 | 0.5% - 1.0% |
| Day Trading | 5 | 1.5% - 2.0% |
| Swing Trading | 7-10 | 2.5% - 3.5% |
| Position Trading | 15-20 | 4.0% - 5.0% |

### GapDetectionConfig

Controls gap pattern detection (kicker, f23x, gapper):

```python
GapDetectionConfig(
    enabled=True,               # Enable gap detection
    kicker_threshold=0.005,     # 0.5% gap threshold
    f23x_threshold=0.01,        # 1.0% gap threshold
    gapper_threshold=0.02       # 2.0% gap threshold
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable/disable gap detection |
| `kicker_threshold` | `float` | `0.005` | Minimum gap size (as decimal) for kicker patterns |
| `f23x_threshold` | `float` | `0.01` | Minimum gap size for f23x patterns |
| `gapper_threshold` | `float` | `0.02` | Minimum gap size for gapper patterns |

### TargetConfig

Controls target price calculation for trading signals:

```python
TargetConfig(
    upper_bound="higher_high",    # Boundary for long targets
    lower_bound="lower_low",      # Boundary for short targets
    merge_threshold_pct=0.02,     # Merge targets within 2%
    max_targets=3                 # Maximum targets per signal
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `upper_bound` | `str` | `"higher_high"` | Boundary for long signals: `"higher_high"` or `"lower_high"` |
| `lower_bound` | `str` | `"lower_low"` | Boundary for short signals: `"lower_low"` or `"higher_low"` |
| `merge_threshold_pct` | `float` | `0.02` | Merge middle targets within threshold (first/last never merged) |
| `max_targets` | `int \| None` | `None` | Maximum targets per signal (`None` = unlimited) |

!!! important "Target Merging Behavior"
    The `merge_threshold_pct` only affects **middle targets**. The first and last targets are always preserved to maintain entry and final exit levels.

## Factory Helper Methods

### Get Supported Timeframes

```python
from thestrat import Factory

# Get list of all supported timeframe strings
timeframes = Factory.get_supported_timeframes()
print(timeframes)
# ['1min', '5min', '15min', '30min', '1h', '4h', '6h', '12h', '1d', '1w', '1m', '1q', '1y']
```

### Get Supported Asset Classes

```python
from thestrat import Factory

# Get list of all supported asset classes
asset_classes = Factory.get_supported_asset_classes()
print(asset_classes)
# ['crypto', 'equities', 'fx']
```

## Complete Configuration Examples

### Minimal Configuration

```python
from thestrat import Factory
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig
)

# Simplest possible configuration with all defaults
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5min"]
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(timeframes=["all"])
        ]
    )
)

pipeline = Factory.create_all(config)
```

### Multi-Timeframe with Custom Settings

```python
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig, SwingPointsConfig, TargetConfig
)

# Advanced multi-timeframe configuration
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5min", "15min", "1h", "1d"],
        asset_class="equities",
        timezone="US/Eastern",
        session_start="09:30"
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            # Aggressive short-term settings
            TimeframeItemConfig(
                timeframes=["5min"],
                swing_points=SwingPointsConfig(window=3, threshold=1.0),
                target_config=TargetConfig(max_targets=5, merge_threshold_pct=0.01)
            ),
            # Balanced medium-term settings
            TimeframeItemConfig(
                timeframes=["15min"],
                swing_points=SwingPointsConfig(window=5, threshold=2.0),
                target_config=TargetConfig(max_targets=3, merge_threshold_pct=0.02)
            ),
            # Conservative long-term settings
            TimeframeItemConfig(
                timeframes=["1h", "1d"],
                swing_points=SwingPointsConfig(window=10, threshold=3.5),
                target_config=TargetConfig(max_targets=2, merge_threshold_pct=0.03)
            )
        ]
    )
)

pipeline = Factory.create_all(config)
```

### Asset-Specific Configurations

=== "US Equities"
    ```python
    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["5min", "15min", "1h"],
            asset_class="equities",
            timezone="US/Eastern",
            session_start="09:30"  # Market open
        ),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=5, threshold=2.0)
                )
            ]
        )
    )
    ```

=== "Crypto (24/7)"
    ```python
    # Note: Crypto support is illustrative - not actively tested
    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["1h", "4h", "1d"],
            asset_class="crypto"
            # timezone automatically forced to UTC
            # hour_boundary automatically True
        ),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=6, threshold=3.0)
                )
            ]
        )
    )
    ```

=== "Forex (24/5)"
    ```python
    # Note: FX support is illustrative - not actively tested
    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["4h", "1d"],
            asset_class="fx"
            # timezone automatically forced to UTC
            # hour_boundary automatically True
        ),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=4, threshold=0.5)
                )
            ]
        )
    )
    ```

## Validation and Error Handling

All configuration uses Pydantic v2 validation with detailed error messages:

```python
from pydantic import ValidationError

try:
    config = AggregationConfig(
        target_timeframes=["5m"],  # Invalid timeframe string
        asset_class="stocks"        # Invalid asset class
    )
except ValidationError as e:
    print(e)
    # Shows detailed validation errors:
    # - Invalid timeframe: must be one of [1min, 5min, 15min, ...]
    # - Invalid asset_class: must be one of [crypto, equities, fx]
```

## Next Steps

- [Quick Start Guide](quickstart.md) - See configuration in action
- [Examples](examples.md) - Advanced configuration patterns
- [Asset Classes](asset-classes.md) - Asset-specific behavior details
- [API Reference](../reference/index.md) - Complete API documentation
