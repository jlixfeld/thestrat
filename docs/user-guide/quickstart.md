# Quick Start

Get up and running with TheStrat in just a few minutes. This guide assumes you have already [installed](installation.md) the module.

## Basic Workflow

TheStrat follows a simple workflow:

1. **Configure** your components using the Factory pattern
2. **Aggregate** your data to the desired timeframe
3. **Analyze** with TheStrat indicators
4. **Extract** signals and insights

## Data Format Requirements

TheStrat requires specific columns in your input data. All data must include a `timeframe` column that specifies the timeframe of each data point.

### Required Columns

**Required columns (all mandatory):**
- `timestamp` - datetime column for each bar/candle
- `timeframe` - timeframe of the data (e.g., '1min', '5min', '1h')
- `open` - opening price (float, > 0)
- `high` - highest price (float, > 0)
- `low` - lowest price (float, > 0)
- `close` - closing price (float, > 0)

**Optional columns:**
- `symbol` - trading symbol/ticker (string)
- `volume` - trading volume (float, ≥ 0)

### Data Format Example

All input data **must** include the `timeframe` column:

```python
# Example input data
data = pd.DataFrame({
    'timestamp': [...],  # datetime
    'timeframe': ['1min', '1min', '5min', '5min', ...],  # MANDATORY
    'open': [...],       # float
    'high': [...],       # float
    'low': [...],        # float
    'close': [...],      # float
    'symbol': [...],     # optional string
    'volume': [...]      # optional float
})
```

**Data type support:**
- Accepts both Polars and Pandas DataFrames (auto-converts Pandas to Polars)
- Timestamps can be timezone-naive (converted using config timezone) or timezone-aware
- Minimum 2 data points required

## Your First TheStrat Analysis

Let's start with a complete example using sample market data:

```python
from pandas import DataFrame as PandasDataFrame
from thestrat import Factory
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig, SwingPointsConfig
)

# Sample OHLCV data (1-minute bars)
sample_data = PandasDataFrame({
    'timestamp': pd.date_range('2024-01-01 09:30', periods=100, freq='1min'),
    'open': [100.0] * 100,
    'high': [101.0] * 100,
    'low': [99.0] * 100,
    'close': [100.5] * 100,
    'volume': [1000] * 100,
    'timeframe': ['1min'] * 100
})

# Configure TheStrat components with Pydantic models
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5min"],  # Aggregate to 5-minute bars (now supports multiple)
        asset_class="equities",      # US equity market
        timezone="US/Eastern"        # Eastern timezone
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["all"],    # Apply to all target timeframes
                swing_points=SwingPointsConfig(
                    window=5,            # 5-period swing detection
                    threshold=2.0        # 2% threshold for significance
                )
            )
        ]
    )
)

# Create components using Factory
pipeline = Factory.create_all(config)

# Process the data - now returns normalized output with timeframe column
aggregated_data = pipeline["aggregation"].process(sample_data)
analyzed_data = pipeline["indicators"].process(aggregated_data)

print(f"Original bars: {len(sample_data)}")
print(f"Aggregated bars: {len(aggregated_data)}")
print(f"Analysis complete: {len(analyzed_data)} bars with TheStrat indicators")
print(f"Timeframes processed: {analyzed_data['timeframe'].unique()}")
```

## Step-by-Step Breakdown

### Step 1: Configure Components

The Factory pattern centralizes configuration:

```python
from thestrat import Factory
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig, SwingPointsConfig
)

# Minimal configuration using models
simple_config = FactoryConfig(
    aggregation=AggregationConfig(target_timeframes=["5min"]),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(timeframes=["all"])
        ]
    )
)

# Full configuration with all options
full_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5min", "15min"],  # Multiple timeframes supported
        asset_class="equities",
        timezone="US/Eastern"
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["5min"],
                swing_points=SwingPointsConfig(window=5, threshold=2.0)
            ),
            TimeframeItemConfig(
                timeframes=["15min"],
                swing_points=SwingPointsConfig(window=7, threshold=3.0)
            )
        ]
    )
)

components = Factory.create_all(full_config)
```

### Step 2: Timeframe Aggregation

Transform your base timeframe data:

```python
# Get the aggregation component
aggregator = components["aggregation"]

# Process your 1-minute data into 5-minute bars
five_min_bars = aggregator.process(one_minute_data)

# The aggregated data maintains OHLCV structure and includes timeframe column
print(five_min_bars.columns)
# ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe']
```

### Step 3: Apply TheStrat Indicators

Analyze market structure:

```python
# Get the indicators component
indicators = components["indicators"]

# Apply TheStrat analysis
analyzed = indicators.process(five_min_bars)

# New columns are added for TheStrat metrics
print(analyzed.columns)
# Original OHLCV + TheStrat indicators like:
# 'scenario', 'higher_high', 'lower_low', 'signal', etc.
```

### Step 4: Extract Insights

Work with the results:

```python
# Find inside bars (scenario == "1")
inside_bars = analyzed.filter(pl.col('scenario') == "1")
print(f"Found {len(inside_bars)} inside bars")

# Find outside bars (scenario == "3")
outside_bars = analyzed.filter(pl.col('scenario') == "3")
print(f"Found {len(outside_bars)} outside bars")

# Find market structure points
higher_highs = len(analyzed['higher_high'].drop_nulls())
lower_lows = len(analyzed['lower_low'].drop_nulls())

print(f"Higher highs: {higher_highs}")
print(f"Lower lows: {lower_lows}")

# Get the latest signals
latest_signals = analyzed.tail(10).select(['timestamp', 'scenario', 'signal'])
print(latest_signals)
```

## Asset Class Examples

Different asset classes require different configurations:

### Crypto (24/7 Trading)

!!! warning "Not Actively Tested"
    Crypto support is illustrative. This configuration is not actively tested or used in production.

```python
crypto_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["1h"],
        asset_class="crypto",
        timezone="UTC"  # Always UTC for crypto
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

crypto_pipeline = Factory.create_all(crypto_config)
```

### Forex (24/5 Trading)

!!! warning "Not Actively Tested"
    Forex support is illustrative. This configuration is not actively tested or used in production.

```python
fx_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["4h"],
        asset_class="fx",
        timezone="UTC"  # Always UTC for FX
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["all"],
                swing_points=SwingPointsConfig(window=5, threshold=1.0)
            )
        ]
    )
)

fx_pipeline = Factory.create_all(fx_config)
```


## Common Patterns

### Multiple Timeframe Analysis

```python
# Analyze multiple timeframes in a single operation using Pydantic models
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5min", "15min", "1h"],  # Process all timeframes together
        asset_class="equities"
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["5min"],
                swing_points=SwingPointsConfig(window=3, threshold=1.5)  # Aggressive for short timeframe
            ),
            TimeframeItemConfig(
                timeframes=["15min", "1h"],
                swing_points=SwingPointsConfig(window=7, threshold=2.5)  # Conservative for longer timeframes
            )
        ]
    )
)

pipeline = Factory.create_all(config)
aggregated = pipeline["aggregation"].process(raw_data)
analyzed = pipeline["indicators"].process(aggregated)

# Filter results by timeframe using the normalized output
for tf in ["5min", "15min", "1h"]:
    tf_data = analyzed.filter(pl.col('timeframe') == tf)
    inside_bars = len(tf_data.filter(pl.col('scenario') == "1"))
    print(f"{tf}: {len(tf_data)} bars, {inside_bars} inside bars")

print(f"Total analysis: {len(analyzed)} bars across {len(analyzed['timeframe'].unique())} timeframes")
```

### Signal Detection

```python
# Custom signal detection
def find_breakouts(data):
    """Find potential breakout signals"""
    breakouts = []

    for i in range(1, len(data)):
        current = data.row(i, named=True)
        previous = data.row(i-1, named=True)

        # Outside bar (scenario == "3") followed by continuation
        if (previous['scenario'] == "3" and
            current['close'] > previous['high']):
            breakouts.append({
                'timestamp': current['timestamp'],
                'type': 'bullish_breakout',
                'price': current['close']
            })

    return breakouts

# Apply to your analyzed data
signals = find_breakouts(analyzed_data)
print(f"Found {len(signals)} breakout signals")
```

## Next Steps

Now that you understand the basics:

1. **Explore [Examples](examples.md)** - More detailed use cases and advanced features
2. **Review [Asset Classes](asset-classes.md)** - Understand market-specific behaviors
3. **Check [API Reference](../reference/index.md)** - Detailed documentation of all methods and parameters

## Common Questions

**Q: What timeframes are supported?**
A: Supported timeframes: 1min, 5min, 15min, 30min, 1h, 4h, 6h, 12h, 1d, 1w, 1m (month), 1q, 1y. These are the only valid timeframe strings.

**Q: Can I use my own data format?**
A: Yes, as long as it has the required columns including the mandatory `timeframe` column. TheStrat accepts both Pandas and Polars DataFrames and will automatically standardize the data.

**Q: What if I have data from multiple symbols?**
A: Include a `symbol` column and TheStrat will process each symbol separately while maintaining proper grouping in the aggregation.

**Q: Why is the timeframe column mandatory?**
A: The timeframe column ensures TheStrat knows exactly what timeframe your data represents, preventing errors and enabling optimal aggregation strategies. This makes the API explicit and prevents silent failures.

**Q: How do I migrate existing data to include the timeframe column?**
A: Add the timeframe column to your existing DataFrames. Here's how:

```python
# If you have existing data without timeframe column
existing_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01 09:30', periods=100, freq='1min'),
    'open': [100.0] * 100,
    'high': [101.0] * 100,
    'low': [99.0] * 100,
    'close': [100.5] * 100,
    'volume': [1000] * 100
})

# Add the timeframe column (1min data in this example)
existing_data['timeframe'] = '1min'

# Or for multiple timeframes, determine based on your data frequency
def determine_timeframe(freq_minutes):
    if freq_minutes == 1:
        return '1min'
    elif freq_minutes == 5:
        return '5min'
    elif freq_minutes == 60:
        return '1h'
    # Add more mappings as needed

# Now your data is ready for TheStrat
pipeline = Factory.create_all(config)
result = pipeline["aggregation"].process(existing_data)
```

**Q: How do I handle missing data?**
A: TheStrat includes built-in handling for gaps and missing bars appropriate to each asset class.

**Q: Can I backtest strategies?**
A: TheStrat provides the analysis foundation. You'll need to combine it with your backtesting framework.
