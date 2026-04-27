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
[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/jlixfeld/thestrat)
[![License](https://img.shields.io/badge/license-Private-red)](https://github.com/jlixfeld/thestrat)

A small Polars-based library for OHLCV timeframe aggregation and Strat bar classification.

> **1.0.0 is a clean rewrite.** The pre-1.0 API (Factory / Indicators / Signals / Schemas) has been removed. Old import paths now raise `ImportError` with a message identifying the consumer as defunct. New surface area is intentionally minimal: an aggregator, a classifier, and a small set of types.

## Features

- **Multi-timeframe OHLCV aggregation** (1m → 5m, 15m, 30m, 1h, 4h, 6h, 12h, 1d, 1w, 1m, 1q, 1y) with optional equity-session offset.
- **Vectorized Strat bar classification** — scenario (1 / 2U / 2D / 3), color, shape (hammer / shooter), in-force flag, plus 3 bars of history.
- **Single-symbol and multi-symbol** classification paths (the latter using `over("symbol")` for per-symbol shifts).
- **Pure functions over Polars DataFrames.** No async, no I/O, no global state.

## Installation

```bash
uv add git+https://github.com/jlixfeld/thestrat.git
```

For development:

```bash
git clone https://github.com/jlixfeld/thestrat.git
cd thestrat
uv sync --extra test --extra dev
```

## Usage

### Aggregation

```python
import polars as pl
from thestrat import TimeframeAggregator

bars = pl.DataFrame({
    "timestamp": [...],
    "open": [...], "high": [...], "low": [...], "close": [...],
    "volume": [...],
})

agg = TimeframeAggregator()
hourly = agg.aggregate(bars, "1h")
hourly_eq = agg.aggregate(bars, "1h", equity_offset=True)  # 9:30, 10:30, ...
daily = agg.aggregate(bars, "1d")
```

### Classification (single symbol)

```python
from thestrat import classify_bars_df

# Input must be sorted by timestamp for a single symbol+timeframe.
classified = classify_bars_df(bars)
# Adds columns: scenario0..3, color0..3, shape, in_force
```

### Classification (multi-symbol)

```python
from thestrat import classify_bars_multi_symbol

# Input must have a `symbol` column and be sorted by (symbol, timestamp).
classified = classify_bars_multi_symbol(bars)
```

### Scalar helpers

```python
from thestrat import classify_bar, classify_color, classify_scenario

color = classify_color(open_price=100.0, close_price=101.0)         # Color.GREEN
scenario = classify_scenario(curr_high=105, curr_low=95,
                             prev_high=104, prev_low=96)            # Scenario.TWO_UP

bar = classify_bar(
    bar={"symbol": "ESM6", "timestamp": ts, "timeframe": "1d",
         "open": 100, "high": 105, "low": 95, "close": 103, "volume": 1000},
    prior=prior_bar_dict_or_None,
)
```

### Types

```python
from thestrat import BarDict, ClassifiedBar, Color, Scenario, Shape, Timeframe
```

## Public API

| Symbol | Where |
|---|---|
| `TimeframeAggregator`, `EQUITY_OFFSET_MINUTES` | `thestrat.aggregator` |
| `classify_bars_df`, `classify_bars_multi_symbol`, `classify_bar`, `classify_color`, `classify_scenario`, `SHAPE_BODY_ZONE` | `thestrat.classifier` |
| `BarDict`, `ClassifiedBar`, `Color`, `Scenario`, `Shape`, `Timeframe` | `thestrat.types` |

All of the above are re-exported from the package root (`from thestrat import ...`).
