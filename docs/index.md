# TheStrat Documentation

[![Version](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/jlixfeld/thestrat/main/pyproject.toml&query=project.version&label=version&color=green)](https://github.com/jlixfeld/thestrat/releases)
[![Tests](https://github.com/jlixfeld/thestrat/actions/workflows/tests.yml/badge.svg)](https://github.com/jlixfeld/thestrat/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jlixfeld/c383059dafef5a6c070532174f3f0ba8/raw/coverage.json)](https://github.com/jlixfeld/thestrat/actions/workflows/coverage.yml)
[![Documentation](https://github.com/jlixfeld/thestrat/actions/workflows/docs.yml/badge.svg)](https://jlixfeld.github.io/thestrat/)
[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://github.com/jlixfeld/thestrat)
[![License](https://img.shields.io/badge/license-Private-red)](https://github.com/jlixfeld/thestrat)

A Python module for financial data aggregation and technical analysis using **#TheStrat** methodology.

!!! info "Primary Focus: US Equities"
    This library is primarily developed and tested for **US Equities** analysis. While crypto, forex, and futures are supported via configuration, they are not actively tested or used in production. All examples and documentation focus on US Equities use cases.

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

US Equities (primary focus), with additional support for crypto and FX timezone alignment

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
        target_timeframes=["5min", "15min"],  # Multiple timeframes supported
        asset_class="equities",
        timezone="US/Eastern"
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["5min"],
                swing_points=SwingPointsConfig(window=3, threshold=1.5)  # Short-term settings
            ),
            TimeframeItemConfig(
                timeframes=["15min"],
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

=== "Equities (Primary)"
    - **US Equities** (actively tested and used)
    - Session-aligned aggregation via configurable `session_start` offset
    - Configurable timezones (default: US/Eastern)
    - Note: No explicit pre/post-market gating or holiday calendars

=== "Crypto (Supported)"
    - 24/7 trading support
    - UTC timezone enforcement
    - Continuous aggregation
    - **Not actively tested** - treat as illustrative

=== "Forex (Supported)"
    - 24/5 alignment with UTC
    - UTC timezone enforcement
    - Weekend gaps appear in price data as-is
    - **Not actively tested** - treat as illustrative


## Getting Started

Ready to implement #TheStrat in your trading system?

[Get Started with Installation](user-guide/installation.md){ .md-button .md-button--primary }
[View API Reference](reference/index.md){ .md-button }

## Project Status

This project is under active development with comprehensive test coverage and strict code quality standards.
