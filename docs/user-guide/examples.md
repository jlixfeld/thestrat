# Examples

This section provides comprehensive examples of using TheStrat for various trading scenarios and asset classes.

## Basic Examples

### Simple 5-Minute Analysis

```python
from pandas import DataFrame as PandasDataFrame
from thestrat import Factory
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig, SwingPointsConfig
)

# Sample market data
data = PandasDataFrame({
    'timestamp': pd.date_range('2024-01-01 09:30', periods=300, freq='1min'),
    'open': [100 + i*0.1 for i in range(300)],
    'high': [100.5 + i*0.1 for i in range(300)],
    'low': [99.5 + i*0.1 for i in range(300)],
    'close': [100.2 + i*0.1 for i in range(300)],
    'volume': [1000 + i*10 for i in range(300)]
})

# Configure for 5-minute equity analysis with Pydantic models
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5m"],
        asset_class="equities",
        timezone="US/Eastern"
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

# Process the data
pipeline = Factory.create_all(config)
aggregated = pipeline["aggregation"].process(data)
analyzed = pipeline["indicators"].process(aggregated)

# Display results
print(f"Converted {len(data)} 1-minute bars to {len(aggregated)} 5-minute bars")
print(f"Found {analyzed['inside_bar'].sum()} inside bars")
print(f"Found {analyzed['outside_bar'].sum()} outside bars")
```

### Multi-Timeframe Analysis

```python
from thestrat import Factory
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig, SwingPointsConfig
)
from pandas import DataFrame as PandasDataFrame

def analyze_multiple_timeframes(data, timeframes=['5m', '15m', '1h']):
    """Analyze data across multiple timeframes using Pydantic models."""
    # Single configuration for multiple timeframes using models
    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=timeframes,  # Process all timeframes together
            asset_class="equities",
            timezone="US/Eastern"
        ),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["5m"],
                    swing_points=SwingPointsConfig(window=3, threshold=1.5)  # Short-term settings
                ),
                TimeframeItemConfig(
                    timeframes=["15m", "1h"],
                    swing_points=SwingPointsConfig(window=7, threshold=2.5)  # Long-term settings
                )
            ]
        )
    )

    # Single pipeline processes all timeframes
    pipeline = Factory.create_all(config)
    aggregated = pipeline["aggregation"].process(data)
    analyzed = pipeline["indicators"].process(aggregated)

    # Extract results by timeframe from normalized output
    results = {}
    for tf in timeframes:
        tf_data = analyzed[analyzed['timeframe'] == tf]
        results[tf] = {
            'data': tf_data,
            'inside_bars': tf_data['inside_bar'].sum(),
            'outside_bars': tf_data['outside_bar'].sum(),
            'higher_highs': len(tf_data['higher_high'].drop_nulls()),
            'lower_lows': len(tf_data['lower_low'].drop_nulls())
        }

    return results, analyzed  # Return both summary and full data

# Use with your data
multi_tf_analysis, full_data = analyze_multiple_timeframes(sample_data)

# Display summary
for tf, result in multi_tf_analysis.items():
    print(f"{tf}: {result['inside_bars']} inside, {result['outside_bars']} outside")

print(f"Total processed: {len(full_data)} bars across {len(full_data['timeframe'].unique())} timeframes")
```

## Asset Class Specific Examples

### Cryptocurrency (24/7 Trading)

```python
# Bitcoin/crypto configuration with Pydantic models
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
                swing_points=SwingPointsConfig(
                    window=6,  # Slightly larger window for hourly
                    threshold=3.0  # Higher threshold for crypto volatility
                )
            )
        ]
    )
)

crypto_pipeline = Factory.create_all(crypto_config)

# Process crypto data (note: no market hours restrictions)
crypto_analyzed = crypto_pipeline["aggregation"].process(btc_data)
crypto_signals = crypto_pipeline["indicators"].process(crypto_analyzed)

print(f"Crypto analysis: 24/7 trading, {len(crypto_signals)} hourly bars")
```

### Forex (24/5 Trading)

```python
# EUR/USD analysis with Pydantic models
fx_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["4h"],
        asset_class="fx",
        timezone="UTC"
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["all"],
                swing_points=SwingPointsConfig(
                    window=4,
                    threshold=0.5  # Lower threshold for FX (measured in %)
                )
            )
        ]
    )
)

fx_pipeline = Factory.create_all(fx_config)

# FX data processing handles weekend gaps automatically
eurusd_aggregated = fx_pipeline["aggregation"].process(eurusd_1m_data)
eurusd_analyzed = fx_pipeline["indicators"].process(eurusd_aggregated)

# Find major market structure points
higher_highs = len(eurusd_analyzed['higher_high'].drop_nulls())
lower_lows = len(eurusd_analyzed['lower_low'].drop_nulls())
print(f"Found {higher_highs} HH and {lower_lows} LL in EUR/USD")
```


## Multi-Timeframe Examples

### Single Request Multi-Timeframe Analysis

```python
from thestrat import Factory

# Process multiple timeframes with different configurations using Pydantic models
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5m", "15m", "1h", "1d"],  # All timeframes together
        asset_class="equities",
        timezone="US/Eastern"
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["5m"],  # Short-term aggressive settings
                swing_points=SwingPointsConfig(window=3, threshold=1.0)
            ),
            TimeframeItemConfig(
                timeframes=["15m"],  # Medium-term balanced settings
                swing_points=SwingPointsConfig(window=5, threshold=2.0)
            ),
            TimeframeItemConfig(
                timeframes=["1h", "1d"],  # Long-term conservative settings
                swing_points=SwingPointsConfig(window=10, threshold=3.0)
            )
        ]
    )
)

pipeline = Factory.create_all(config)
aggregated = pipeline["aggregation"].process(market_data)
analyzed = pipeline["indicators"].process(aggregated)

# Extract results for each timeframe from normalized output
for tf in ["5m", "15m", "1h", "1d"]:
    tf_data = analyzed[analyzed['timeframe'] == tf]
    print(f"{tf}: {len(tf_data)} bars, {tf_data['inside_bar'].sum()} inside bars")

print(f"Total: {len(analyzed)} bars across {len(analyzed['timeframe'].unique())} timeframes")
```

## Swing Point Analysis

### Understanding Swing Point Detection

Swing points are critical for identifying market structure in TheStrat methodology. The implementation uses precise peak/valley detection with configurable parameters.

```python
from thestrat import Factory
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig, SwingPointsConfig
)
from pandas import DataFrame as PandasDataFrame

def analyze_swing_points(data):
    """Demonstrate swing point detection with different configurations."""

    # Configuration with detailed swing point settings
    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["5m"],
            asset_class="equities",
            timezone="US/Eastern"
        ),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(
                        window=5,        # Look 5 bars back and ahead
                        threshold=2.0    # Require 2% price change to confirm
                    )
                )
            ]
        )
    )

    # Process the data
    pipeline = Factory.create_all(config)
    aggregated = pipeline["aggregation"].process(data)
    analyzed = pipeline["indicators"].process(aggregated)

    # Market structure analysis
    higher_highs = len(analyzed['higher_high'].drop_nulls())
    lower_highs = len(analyzed['lower_high'].drop_nulls())
    higher_lows = len(analyzed['higher_low'].drop_nulls())
    lower_lows = len(analyzed['lower_low'].drop_nulls())

    print(f"Higher highs: {higher_highs} (bullish structure)")
    print(f"Lower lows: {lower_lows} (bearish structure)")
    print(f"Lower lows: {len(lower_lows)} (bearish structure)")

    return analyzed

# Example usage with trending data
trending_data = PandasDataFrame({
    'timestamp': pd.date_range('2024-01-01 09:30', periods=100, freq='5min'),
    'open': [100 + i*0.5 + (i%10)*0.2 for i in range(100)],    # Trending up with oscillations
    'high': [101 + i*0.5 + (i%10)*0.3 for i in range(100)],
    'low': [99 + i*0.5 + (i%10)*0.1 for i in range(100)],
    'close': [100.5 + i*0.5 + (i%10)*0.25 for i in range(100)],
    'volume': [1000 + i*10 for i in range(100)]
})

results = analyze_swing_points(trending_data)
```

### Swing Point Configuration Strategies

Different market conditions and trading styles require different swing point settings:

```python
def compare_swing_configurations(data):
    """Compare different swing point configurations."""

    configurations = [
        ("Scalping", SwingPointsConfig(window=3, threshold=0.5)),     # Very sensitive
        ("Day Trading", SwingPointsConfig(window=5, threshold=1.5)),  # Balanced
        ("Swing Trading", SwingPointsConfig(window=10, threshold=3.0)), # Conservative
        ("Position Trading", SwingPointsConfig(window=20, threshold=5.0)) # Very conservative
    ]

    results = {}

    for strategy_name, swing_config in configurations:
        config = FactoryConfig(
            aggregation=AggregationConfig(
                target_timeframes=["5m"],
                asset_class="equities"
            ),
            indicators=IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=swing_config
                    )
                ]
            )
        )

        pipeline = Factory.create_all(config)
        aggregated = pipeline["aggregation"].process(data)
        analyzed = pipeline["indicators"].process(aggregated)

        # Count market structure points detected
        structure_count = (len(analyzed['higher_high'].drop_nulls()) +
                          len(analyzed['lower_high'].drop_nulls()) +
                          len(analyzed['higher_low'].drop_nulls()) +
                          len(analyzed['lower_low'].drop_nulls()))

        results[strategy_name] = {
            'config': swing_config,
            'swing_points': swing_count,
            'frequency': f"{swing_count/len(analyzed)*100:.1f}% of bars"
        }

        print(f"{strategy_name}: {swing_count} swing points ({results[strategy_name]['frequency']})")

    return results

# Compare configurations
config_results = compare_swing_configurations(trending_data)
```

### Market Structure Trend Analysis

Understanding the relationship between swing highs and lows reveals market trends:

```python
def analyze_market_structure_trend(analyzed_data):
    """Analyze trend direction using market structure."""

    # Get market structure data
    structure_data = analyzed_data.filter(
        (analyzed_data['higher_high'].is_not_null()) |
        (analyzed_data['lower_low'].is_not_null())
    ).sort('timestamp')

    if len(swing_points) < 4:
        return "Insufficient swing points for trend analysis"

    # Count recent structure patterns
    recent_data = analyzed_data.tail(50)  # Last 50 bars

    hh_count = len(recent_data['higher_high'].drop_nulls())
    hl_count = len(recent_data['higher_low'].drop_nulls())
    lh_count = len(recent_data['lower_high'].drop_nulls())
    ll_count = len(recent_data['lower_low'].drop_nulls())

    bullish_signals = hh_count + hl_count
    bearish_signals = lh_count + ll_count

    print(f"Recent Market Structure (last 50 bars):")
    print(f"  Higher Highs: {hh_count}")
    print(f"  Higher Lows: {hl_count}")
    print(f"  Lower Highs: {lh_count}")
    print(f"  Lower Lows: {ll_count}")
    print(f"  Bullish signals: {bullish_signals}")
    print(f"  Bearish signals: {bearish_signals}")

    if bullish_signals > bearish_signals * 1.5:
        trend = "Strong Uptrend"
    elif bearish_signals > bullish_signals * 1.5:
        trend = "Strong Downtrend"
    elif bullish_signals > bearish_signals:
        trend = "Weak Uptrend"
    elif bearish_signals > bullish_signals:
        trend = "Weak Downtrend"
    else:
        trend = "Sideways/Consolidation"

    print(f"  Trend Assessment: {trend}")
    return trend

# Analyze the trend
trend_assessment = analyze_market_structure_trend(results)
```

### Performance Considerations

TheStrat's swing point detection is fully vectorized for optimal performance:

```python
import time

def benchmark_swing_detection(data_size=10000):
    """Benchmark swing point detection performance."""

    # Generate large dataset
    large_data = PandasDataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=data_size, freq='1min'),
        'open': [100 + i*0.01 + (i%100)*0.1 for i in range(data_size)],
        'high': [100.5 + i*0.01 + (i%100)*0.15 for i in range(data_size)],
        'low': [99.5 + i*0.01 + (i%100)*0.05 for i in range(data_size)],
        'close': [100.2 + i*0.01 + (i%100)*0.12 for i in range(data_size)],
        'volume': [1000 + i for i in range(data_size)]
    })

    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["5m"],
            asset_class="equities"
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

    pipeline = Factory.create_all(config)

    # Benchmark aggregation
    start_time = time.time()
    aggregated = pipeline["aggregation"].process(large_data)
    agg_time = time.time() - start_time

    # Benchmark indicators (including swing points)
    start_time = time.time()
    analyzed = pipeline["indicators"].process(aggregated)
    indicator_time = time.time() - start_time

    print(f"Performance Benchmark ({data_size:,} input rows):")
    print(f"  Aggregation: {agg_time:.3f}s ({len(large_data)/agg_time:,.0f} rows/sec)")
    print(f"  Indicators: {indicator_time:.3f}s ({len(aggregated)/indicator_time:,.0f} rows/sec)")
    print(f"  Total: {agg_time + indicator_time:.3f}s")
    print(f"  Output: {len(analyzed)} bars with full indicator analysis")

# Run performance benchmark
benchmark_swing_detection(10000)
```

### Cross-Timeframe Signal Correlation

```python
def analyze_cross_timeframe_signals(data):
    """Analyze signal correlation across multiple timeframes."""
    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["5m", "15m", "1h"],
            asset_class="equities"
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

    pipeline = Factory.create_all(config)
    aggregated = pipeline["aggregation"].process(data)
    analyzed = pipeline["indicators"].process(aggregated)

    # Find synchronized signals across timeframes
    synchronized_signals = []

    # Get latest bar for each timeframe
    latest_by_tf = {}
    for tf in ["5m", "15m", "1h"]:
        tf_data = analyzed[analyzed['timeframe'] == tf]
        if len(tf_data) > 0:
            latest_by_tf[tf] = tf_data.iloc[-1]

    # Check for signal alignment
    if all(bar.get('outside_bar', False) for bar in latest_by_tf.values()):
        synchronized_signals.append({
            'type': 'multi_timeframe_breakout',
            'timeframes': list(latest_by_tf.keys()),
            'timestamp': list(latest_by_tf.values())[0]['timestamp']
        })

    return synchronized_signals, analyzed

# Example usage
signals, full_analysis = analyze_cross_timeframe_signals(sample_data)
print(f"Found {len(signals)} synchronized signals across multiple timeframes")
```

## Advanced Analysis Examples

### Custom Signal Detection

```python
def detect_strat_patterns(data):
    """Detect common TheStrat patterns."""
    patterns = []

    for i in range(2, len(data)):
        current = data.iloc[i]
        prev1 = data.iloc[i-1]
        prev2 = data.iloc[i-2]

        # Inside bar followed by breakout (2-1-2 Continuation)
        if (prev2['outside_bar'] and
            prev1['inside_bar'] and
            current['close'] > prev2['high']):
            patterns.append({
                'timestamp': current['timestamp'],
                'pattern': '2-1-2_bullish_continuation',
                'entry_price': prev2['high'],
                'target': current['close'] + (current['close'] - prev2['low']) * 0.5
            })

        # Outside bar reversal
        if (current['outside_bar'] and
            prev1['close'] > prev1['open'] and  # Previous bar was bullish
            current['close'] < current['open']):  # Current bar is bearish
            patterns.append({
                'timestamp': current['timestamp'],
                'pattern': 'outside_bar_reversal',
                'entry_price': current['low'],
                'stop_loss': current['high']
            })

    return patterns

# Apply pattern detection
patterns = detect_strat_patterns(analyzed_data)
print(f"Detected {len(patterns)} TheStrat patterns")

# Display recent patterns
for pattern in patterns[-5:]:
    print(f"{pattern['timestamp']}: {pattern['pattern']} @ {pattern['entry_price']}")
```

### Risk Management Integration

```python
def calculate_position_sizes(signals, account_balance, risk_percent=2.0):
    """Calculate position sizes based on TheStrat signals."""
    positions = []

    for signal in signals:
        if 'entry_price' in signal and 'stop_loss' in signal:
            # Calculate risk per share
            risk_per_share = abs(signal['entry_price'] - signal['stop_loss'])

            # Calculate position size
            risk_amount = account_balance * (risk_percent / 100)
            position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

            positions.append({
                **signal,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'risk_per_share': risk_per_share
            })

    return positions

# Example usage
account_balance = 100000  # $100k account
risk_per_trade = 2.0      # 2% risk per trade

sized_positions = calculate_position_sizes(patterns, account_balance, risk_per_trade)

for pos in sized_positions:
    if pos['position_size'] > 0:
        print(f"Signal: {pos['pattern']}")
        print(f"Entry: ${pos['entry_price']:.2f}")
        print(f"Size: {pos['position_size']} shares")
        print(f"Risk: ${pos['risk_amount']:.2f}")
        print("---")
```

### Real-Time Analysis Simulation

```python
import time
from datetime import datetime

def simulate_real_time_analysis(historical_data, interval_seconds=60):
    """Simulate real-time TheStrat analysis with Pydantic models."""

    config = FactoryConfig(
        aggregation=AggregationConfig(target_timeframes=["5m"], asset_class="equities"),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=5, threshold=2.0)
                )
            ]
        )
    )

    pipeline = Factory.create_all(config)

    # Simulate streaming data
    for i in range(50, len(historical_data), 5):  # Add 5 bars at a time
        current_data = historical_data.iloc[:i]

        # Process latest data
        aggregated = pipeline["aggregation"].process(current_data)
        analyzed = pipeline["indicators"].process(aggregated)

        # Check for new signals (last bar)
        if len(analyzed) > 0:
            latest = analyzed.iloc[-1]

            if latest['inside_bar']:
                print(f"{datetime.now()}: Inside bar detected @ {latest['close']:.2f}")
            elif latest['outside_bar']:
                print(f"{datetime.now()}: Outside bar detected @ {latest['close']:.2f}")

            # Check for market structure changes
            if latest.get('higher_high') is not None:
                print(f"{datetime.now()}: Higher High @ {latest['higher_high']:.2f}")
            elif latest.get('lower_low') is not None:
                print(f"{datetime.now()}: Lower Low @ {latest['lower_low']:.2f}")

        time.sleep(interval_seconds)

# Run simulation (comment out for docs)
# simulate_real_time_analysis(sample_data, interval_seconds=2)
```

## Performance Optimization Examples

### Batch Processing

```python
def batch_process_symbols(symbol_data_dict, config_template):
    """Process multiple symbols efficiently with new API."""
    results = {}

    # Create pipeline once - supports multiple timeframes per symbol
    pipeline = Factory.create_all(config_template)

    for symbol, data in symbol_data_dict.items():
        try:
            # Process each symbol - now handles multiple timeframes
            aggregated = pipeline["aggregation"].process(data)
            analyzed = pipeline["indicators"].process(aggregated)

            # Store results with timeframe breakdown
            results[symbol] = {
                'data': analyzed,
                'timeframes': analyzed['timeframe'].unique().tolist(),
                'inside_bars': analyzed['inside_bar'].sum(),
                'outside_bars': analyzed['outside_bar'].sum(),
                'last_price': analyzed.iloc[-1]['close'],
                'total_bars': len(analyzed)
            }

            print(f"Processed {symbol}: {len(analyzed)} bars across {len(analyzed['timeframe'].unique())} timeframes")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            results[symbol] = None

    return results

# Example usage
symbols_data = {
    'AAPL': aapl_data,
    'MSFT': msft_data,
    'GOOGL': googl_data
}

batch_results = batch_process_symbols(symbols_data, config)
```

### Memory Efficient Processing

```python
def process_large_dataset(data, chunk_size=1000):
    """Process large datasets in chunks to manage memory with new API."""

    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["5m"],
            asset_class="equities"
        ),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=5)
                )
            ]
        )
    )

    pipeline = Factory.create_all(config)
    results = []

    # Process in chunks
    for start_idx in range(0, len(data), chunk_size):
        end_idx = min(start_idx + chunk_size, len(data))
        chunk = data.iloc[start_idx:end_idx]

        # Include overlap for continuity
        if start_idx > 0:
            overlap = data.iloc[max(0, start_idx-100):start_idx]
            chunk = pd.concat([overlap, chunk])

        # Process chunk
        aggregated = pipeline["aggregation"].process(chunk)
        analyzed = pipeline["indicators"].process(aggregated)

        # Store results (excluding overlap)
        if start_idx > 0:
            analyzed = analyzed.iloc[20:]  # Remove overlap portion

        results.append(analyzed)
        print(f"Processed chunk {start_idx//chunk_size + 1}")

    # Combine results
    final_result = pd.concat(results, ignore_index=True)
    return final_result
```

## Integration Examples

### With Popular Trading Libraries

```python
# Integration with backtrader
import backtrader as bt

class TheStratStrategy(bt.Strategy):
    def __init__(self):
        self.thestrat_config = FactoryConfig(
            aggregation=AggregationConfig(
                target_timeframes=["5m"],
                asset_class="equities"
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
        self.pipeline = Factory.create_all(self.thestrat_config)

    def next(self):
        # Convert backtrader data to DataFrame
        data = self.convert_bt_data()

        # Apply TheStrat analysis
        analyzed = self.pipeline["indicators"].process(
            self.pipeline["aggregation"].process(data)
        )

        # Trading logic based on TheStrat signals
        if analyzed.iloc[-1]['outside_bar'] and not self.position:
            self.buy()
        elif analyzed.iloc[-1]['inside_bar'] and self.position:
            self.close()

# Integration with zipline
from zipline.api import order, record, symbol

def thestrat_zipline_algo(context, data):
    # Get price data
    prices = data.history(symbol('AAPL'), ['open', 'high', 'low', 'close'], 100, '1d')

    # Apply TheStrat with new API
    config = FactoryConfig(
        aggregation=AggregationConfig(target_timeframes=["1d"]),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=5)
                )
            ]
        )
    )
    pipeline = Factory.create_all(config)
    aggregated = pipeline["aggregation"].process(prices.reset_index())
    analyzed = pipeline["indicators"].process(aggregated)

    # Trading decisions
    if analyzed.iloc[-1]['outside_bar']:
        order(symbol('AAPL'), 100)

    record(inside_bars=analyzed['inside_bar'].sum())
```

These examples demonstrate the flexibility and power of TheStrat for various trading scenarios. Adapt the configurations and logic to match your specific trading strategy and requirements.
