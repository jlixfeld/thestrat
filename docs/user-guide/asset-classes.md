# Asset Classes

TheStrat supports multiple asset classes, each with specific market characteristics and trading hours. This guide explains how to configure and work with different asset classes effectively.

!!! info "Primary Focus: US Equities"
    This library is primarily developed and tested for **US Equities** analysis. While crypto, forex, and futures are supported via configuration, they are not actively tested or used in production.

## Overview

Asset classes in TheStrat determine:

- **Trading hours** and session handling
- **Timezone** requirements and defaults
- **Gap handling** for market opens/closes
- **Aggregation behavior** for weekends and holidays

## Supported Asset Classes

TheStrat currently supports three major asset classes: **crypto**, **equities**, and **fx**. Each has been optimized for their specific market characteristics.

### Crypto (24/7 Trading)

Cryptocurrency markets trade continuously without breaks.

**Configuration:**
```python
from thestrat.schemas import FactoryConfig, AggregationConfig, IndicatorsConfig

crypto_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["1h"],
        asset_class="crypto",
        timezone="UTC"  # Always UTC for crypto
    )
)
```

**Characteristics:**
- **Trading Hours**: 24/7/365 continuous
- **Timezone**: UTC (required)
- **Session Handling**: No sessions or gaps
- **Weekend Behavior**: Trades through weekends

**Example Usage:**
```python
from thestrat import Factory
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig, SwingPointsConfig
)

# Bitcoin hourly analysis
btc_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["4h"],
        asset_class="crypto",
        timezone="UTC"
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["all"],
                swing_points=SwingPointsConfig(
                    window=6,           # Longer window for 4h timeframe
                    threshold=4.0       # Higher threshold for crypto volatility
                )
            )
        ]
    )
)

pipeline = Factory.create_all(btc_config)
```

**Best Practices:**
- Use higher volatility thresholds (3-5%)
- Consider larger swing windows due to 24/7 nature
- Include incomplete bars for real-time analysis

---

### Equities (Market Hours)

Traditional stock markets with defined trading sessions.

**Configuration:**
```python
from thestrat.schemas import FactoryConfig, AggregationConfig

equity_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5min"],
        asset_class="equities",
        timezone="US/Eastern"  # NYSE/NASDAQ
    )
)
```

**Characteristics:**
- **Trading Hours**: 9:30 AM - 4:00 PM ET (regular session)
- **Timezone**: US/Eastern (default), configurable
- **Session Handling**: Session-aligned aggregation via configurable session_start offset. No explicit pre/post-market gating or holiday calendars.
- **Weekend Behavior**: No trading weekends/holidays

**Market Sessions:**

TheStrat automatically handles market hours for equities. You can configure different timeframes for different analysis needs:

```python
# Short-term intraday analysis
short_term_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["1m"],
        asset_class="equities",
        timezone="US/Eastern"
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["all"],
                swing_points=SwingPointsConfig(window=3, threshold=1.0)
            )
        ]
    )
)

# Regular session analysis
regular_config = FactoryConfig(
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

# Create pipelines
short_pipeline = Factory.create_all(short_term_config)
regular_pipeline = Factory.create_all(regular_config)
```

**International Equities:**
```python
# London Stock Exchange
lse_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["15m"],
        asset_class="equities",
        timezone="Europe/London"  # 8:00-16:30 GMT
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

# Tokyo Stock Exchange
tse_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["30m"],
        asset_class="equities",
        timezone="Asia/Tokyo"    # 9:00-15:00 JST
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

---

### Forex (24/5 Trading)

Foreign exchange markets trade 24/5 from Sunday 5 PM to Friday 5 PM ET.

**Configuration:**
```python
from thestrat.schemas import FactoryConfig, AggregationConfig

fx_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["1h"],
        asset_class="fx",
        timezone="UTC"  # Always UTC for FX
    )
)
```

**Characteristics:**
- **Trading Hours**: Sun 5 PM - Fri 5 PM ET (24/5)
- **Timezone**: UTC (required)
- **Session Handling**: 24/5 alignment with UTC. Weekend gaps appear in price data as-is; no special stitching logic.
- **Weekend Behavior**: Weekend gaps appear in price data as-is

**Major FX Sessions:**
```python
def analyze_fx_sessions(eurusd_data):
    """Analyze EUR/USD across major FX sessions."""

    sessions = {
        "asian": FactoryConfig(
            aggregation=AggregationConfig(
                target_timeframes=["1h"],
                asset_class="fx",
                timezone="UTC"  # Asian session: 10 PM - 8 AM UTC
            )
        ),
        "london": FactoryConfig(
            aggregation=AggregationConfig(
                target_timeframes=["30min"],
                asset_class="fx",
                timezone="UTC"  # London session: 7 AM - 4 PM UTC
            )
        ),
        "newyork": FactoryConfig(
            aggregation=AggregationConfig(
                target_timeframes=["15min"],
                asset_class="fx",
                timezone="UTC"  # New York session: 12 PM - 9 PM UTC
            )
        )
    }

    results = {}
    for session, config in sessions.items():
        pipeline = Factory.create_all(config)
        analyzed = pipeline["indicators"].process(
            pipeline["aggregation"].process(eurusd_data)
        )
        results[session] = analyzed

    return results
```

**Currency-Specific Examples:**
```python
# Major pairs with different characteristics
pairs = {
    "EURUSD": {"threshold": 0.3, "window": 5},   # Lower volatility
    "GBPJPY": {"threshold": 0.8, "window": 4},   # Higher volatility
    "AUDUSD": {"threshold": 0.4, "window": 6},   # Commodity currency
    "USDCAD": {"threshold": 0.5, "window": 5}    # Oil correlation
}

for pair, params in pairs.items():
    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["4h"],
            asset_class="fx",
            timezone="UTC"
        ),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(**params)
                )
            ]
        )
    )
    # Process each pair...
```

---

## Asset Class Comparison

| Asset Class | Trading Hours | Timezone | Gap Handling | Volatility | Recommended Timeframes |
|-------------|---------------|----------|--------------|------------|----------------------|
| **Crypto** | 24/7 | UTC | None | High | 1h, 4h, 1d |
| **Equities** | 9:30-16:00 ET | US/Eastern | Daily gaps | Medium | 1m, 5m, 15m, 1h |
| **Forex** | 24/5 | UTC | Weekend gaps | Medium | 15m, 1h, 4h |

## Advanced Configuration

### Custom Market Configuration

```python
# Define custom market behavior using supported parameters
custom_config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["15m"],  # Use supported timeframe
        asset_class="equities",     # Base on existing class
        timezone="US/Pacific",      # Custom timezone
        session_start="06:30"       # Custom session start
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

### Multi-Asset Portfolio

```python
def analyze_multi_asset_portfolio(assets):
    """Analyze multiple asset classes in one portfolio."""

    results = {}

    for asset_name, (data, asset_class) in assets.items():
        # Get appropriate config for asset class
        base_config = {
            "crypto": {"target_timeframes": ["4h"], "timezone": "UTC", "threshold": 3.0},
            "equities": {"target_timeframes": ["15min"], "timezone": "US/Eastern", "threshold": 1.5},
            "fx": {"target_timeframes": ["1h"], "timezone": "UTC", "threshold": 0.5}
        }

        if asset_class in base_config:
            config = FactoryConfig(
                aggregation=AggregationConfig(
                    asset_class=asset_class,
                    target_timeframes=base_config[asset_class]["target_timeframes"],
                    timezone=base_config[asset_class]["timezone"]
                ),
                indicators=IndicatorsConfig(
                    timeframe_configs=[
                        TimeframeItemConfig(
                            timeframes=["all"],
                            swing_points=SwingPointsConfig(
                                window=5,
                                threshold=base_config[asset_class]["threshold"]
                            )
                        )
                    ]
                )
            )

            pipeline = Factory.create_all(config)
            aggregated = pipeline["aggregation"].process(data)
            analyzed = pipeline["indicators"].process(aggregated)

            results[asset_name] = {
                'asset_class': asset_class,
                'data': analyzed,
                'inside_bars': len(analyzed.filter(pl.col('scenario') == "1")),
                'outside_bars': len(analyzed.filter(pl.col('scenario') == "3"))
            }

    return results

# Example usage
portfolio = {
    'BTC': (btc_data, 'crypto'),
    'AAPL': (aapl_data, 'equities'),
    'EURUSD': (eurusd_data, 'fx')
}

portfolio_analysis = analyze_multi_asset_portfolio(portfolio)
```

## Calendar Period Aggregation

### Timestamp Alignment for Calendar Periods

When aggregating to calendar-based periods (monthly, quarterly, yearly), TheStrat aligns windows to actual calendar boundaries (1st of month, start of quarter, Jan 1) while applying the session start offset. This ensures calendar periods are comparable across datasets regardless of their start date.

**Equities Example:**
```python
from thestrat import Factory
from thestrat.schemas import FactoryConfig, AggregationConfig

# Aggregate daily equities data to monthly
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["1m"],  # Monthly
        asset_class="equities",
        timezone="US/Eastern"
    )
)

pipeline = Factory.create_all(config)
result = pipeline["aggregation"].process(daily_data)

# Monthly bars align to 1st of month at 09:30 ET
print(result["timestamp"][0])  # 2023-01-01 09:30:00-05:00
```

**Asset Class Behavior:**

| Asset Class | Session Start | Monthly/Quarterly/Yearly Timestamp |
|-------------|---------------|----------------------------------|
| **Equities** | 09:30 ET | 1st of month/quarter/year at 09:30 ET |
| **Crypto** | 00:00 UTC | 1st of month/quarter/year at 00:00 UTC |
| **Forex** | 00:00 UTC | 1st of month/quarter/year at 00:00 UTC |

**Why This Matters:**

Calendar periods anchor to actual calendar boundaries, ensuring:
- **Monthly bars** always start on the 1st of the month
- **Quarterly bars** always start on Jan 1, Apr 1, Jul 1, or Oct 1
- **Yearly bars** always start on Jan 1
- **Consistency** across datasets with different start dates

The session start offset is applied to these calendar boundaries, so:
- Monthly equity bars: **1st of month at 09:30 ET**
- Monthly crypto bars: **1st of month at 00:00 UTC**

**Supported Calendar Periods:**
- `"1m"` - Monthly (anchors to 1st of month)
- `"1q"` - Quarterly (anchors to Jan/Apr/Jul/Oct 1)
- `"1y"` - Yearly (anchors to Jan 1)

## Best Practices by Asset Class

### Crypto
- Use UTC timezone exclusively
- Higher volatility thresholds (3-5%)
- Consider 24/7 nature in signal interpretation
- Include incomplete bars for real-time analysis

### Equities
- Respect market hours and gaps
- Lower volatility thresholds (1-2%)
- Session-aligned aggregation via configurable session_start offset
- Account for earnings and announcement gaps

### Forex
- Use UTC timezone for consistency
- Medium volatility thresholds (0.5-1%)
- Consider major session overlaps
- Handle weekend gaps appropriately

Choose the asset class configuration that matches your data and trading requirements. The framework handles the complex details of market hours, timezone conversions, and gap handling automatically.
