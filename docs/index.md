# TheStrat Documentation

[![Tests](https://github.com/jlixfeld/thestrat/actions/workflows/tests.yml/badge.svg)](https://github.com/jlixfeld/thestrat/actions/workflows/tests.yml)
[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://github.com/jlixfeld/thestrat)

A Python module for financial data aggregation and technical analysis using **#TheStrat** methodology.

## Overview

TheStrat provides a comprehensive framework for implementing the #TheStrat trading methodology in Python. It offers high-performance timeframe aggregation, complete technical indicators, and robust support for multiple asset classes.

## Key Features

:material-trending-up:{ .lg .middle } **Multi-Timeframe Aggregation**

OHLCV data aggregation across multiple timeframes simultaneously with timezone handling

---

:material-chart-line:{ .lg .middle } **#TheStrat Indicators**

Complete implementation of TheStrat technical indicators with per-timeframe configurations

---

:material-finance:{ .lg .middle } **Multi-Asset Support**

Crypto, Equities, and FX with appropriate market hours and timezone handling

---

:material-cog:{ .lg .middle } **Factory Pattern**

Clean component creation and configuration management

---

:material-lightning-bolt:{ .lg .middle } **High Performance**

Vectorized operations using Polars and Pandas for optimal speed

---

:material-check-all:{ .lg .middle } **Comprehensive Testing**

High test coverage with 190+ tests ensuring reliability

## Quick Example

```python title="Basic TheStrat Usage with Pydantic Models"
from thestrat import Factory
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig, SwingPointsConfig
)

# Configure your pipeline with Pydantic models
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5m", "15m"],  # Multiple timeframes supported
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
                timeframes=["15m"],
                swing_points=SwingPointsConfig(window=7, threshold=2.5)  # Long-term settings
            )
        ]
    )
)

# Create and use components
pipeline = Factory.create_all(config)
aggregated = pipeline["aggregation"].process(market_data)
analyzed = pipeline["indicators"].process(aggregated)

print(f"Processed {len(analyzed)} bars with TheStrat indicators")
print(f"Timeframes: {analyzed['timeframe'].unique()}")
```

## Core Components

| Component | Purpose | Features |
|-----------|---------|----------|
| **Aggregation** | OHLCV timeframe aggregation | Timezone handling, simultaneous multi-timeframe processing |
| **Indicators** | TheStrat technical indicators | Inside/Outside bars, Swing points, per-timeframe configurations |
| **Factory** | Component creation | Validation, configuration management |
| **Schemas** | Configuration models | Pydantic validation, comprehensive documentation |

## Supported Markets

=== "Crypto"
    - 24/7 trading
    - UTC timezone enforcement
    - Continuous aggregation

=== "Equities"
    - Market hours (9:30-16:00 ET)
    - Configurable timezones
    - Pre/post market handling

=== "Forex"
    - 24/5 trading (Sun 5pm - Fri 5pm ET)
    - UTC timezone enforcement
    - Weekend gap handling


## Getting Started

Ready to implement #TheStrat in your trading system?

[Get Started with Installation](user-guide/installation.md){ .md-button .md-button--primary }
[View API Reference](reference/index.md){ .md-button }

## Project Status

**Version**: 1.0.1 - Production/Stable
**Python**: 3.11+
**License**: Private - All rights reserved
