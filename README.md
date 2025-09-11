# TheStrat

## Important Notice About Terminology and Community

**"The Strat"** refers to the trading methodology developed by the late Rob Smith and fostered by the broader trading community. I make **NO CLAIMS** to ownership of:
- The Strat methodology or concepts
- The terminology "The Strat" 
- Any trading strategies or techniques discussed in the community
- The broader educational content around these concepts

This repository contains **MY IMPLEMENTATION** of software tools related to The Strat concepts. The community and methodology existed long before this code and will continue regardless of this repository.

## Code License: No License (All Rights Reserved)

The **source code** in this repository is provided without a license. This means:

✅ **You CAN:**
- View and study the code
- Learn from my implementation approach
- Create your own completely independent implementation of The Strat concepts
- Use The Strat methodology in your trading (obviously!)
- Participate in The Strat community without any restrictions from me

❌ **You CANNOT:**
- Copy, modify, or distribute this specific code
- Use this code in your own projects without explicit permission
- Redistribute any part of this codebase

## Why This Distinction Matters

I'm sharing my code to contribute to The Strat community, but I'm not providing free tech support for my implementation. If you want to build Strat tools:

1. **Study the concepts** (freely available from Rob Smith and the community)
2. **Build your own implementation** (which you're completely free to do)
3. **Don't copy my code** (but feel free to be inspired by the approach)

## No Support Policy

I don't provide support for this code. The Strat community has many great resources for learning the methodology itself, but I won't be answering questions about my specific implementation.

---

[![Tests](https://github.com/jlixfeld/thestrat/actions/workflows/tests.yml/badge.svg)](https://github.com/jlixfeld/thestrat/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jlixfeld/c383059dafef5a6c070532174f3f0ba8/raw/coverage.json)](https://github.com/jlixfeld/thestrat/actions/workflows/coverage.yml)
[![Documentation](https://github.com/jlixfeld/thestrat/actions/workflows/docs.yml/badge.svg)](https://jlixfeld.github.io/thestrat/)
[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://github.com/jlixfeld/thestrat)

A Python module for financial data aggregation and technical analysis using #TheStrat methodology.

## Features

- **Multi-Timeframe Aggregation**: OHLCV data aggregation across multiple timeframes in a single operation
- **#TheStrat Indicators**: Complete implementation of TheStrat technical indicators with per-timeframe configurations
- **Asset Class Support**: Crypto, Equities, FX
- **Factory Pattern**: Clean component creation and configuration
- **High Performance**: Vectorized operations using Polars

## Quick Start

### Installation

```bash
# Install from GitHub
uv add git+https://github.com/jlixfeld/thestrat.git

# Install in development mode
git clone https://github.com/jlixfeld/thestrat.git
cd thestrat
uv sync --extra test --extra dev
```

### Basic Usage

```python
from thestrat import Factory
from thestrat.schemas import FactoryConfig

# Create components with validated configuration
config = FactoryConfig(
    aggregation={
        "target_timeframes": ["5m"],
        "asset_class": "equities",
    },
    indicators={
        "timeframe_configs": [
            {
                "timeframes": ["all"],  # Apply to all target timeframes
                "swing_points": {
                    "window": 5,
                    "threshold": 2.0  # 2% threshold
                }
            }
        ]
    }
)

pipeline = Factory.create_all(config)

# Process market data - returns normalized output with timeframe column
aggregated = pipeline["aggregation"].process(market_data)
analyzed = pipeline["indicators"].process(aggregated)

print(f"Processed {len(analyzed)} bars with TheStrat indicators")
print(f"Timeframes processed: {analyzed['timeframe'].unique()}")
```
