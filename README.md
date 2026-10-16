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

The current API includes an aggregator, classification functions, session presets and domain types. Pre-1.0 import paths raise `ImportError`; see the [API reference](docs/api.md#legacy-imports).

## Features

- **OHLCV aggregation** to `1min`, `5min`, `15min`, `30min`, `1h`, `4h`, `6h`, `12h`, `1d`, `1w`, `1m`, `1q`, `1y`. `1min` means one minute; `1m` means one calendar month.
- **Vectorized Strat bar classification** — scenario (1 / 2U / 2D / 3), color, shape (hammer / shooter), in-force flag, plus 3 bars of history.
- **Single-symbol and multi-symbol** classification paths (the latter using `over("symbol")` for per-symbol shifts).
- **Functions over Polars DataFrames**, with no network, file or broker I/O. Session presets are available through the exported `SESSIONS` dictionary.

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
from datetime import datetime

import polars as pl
from thestrat import TimeframeAggregator

bars = pl.DataFrame(
    {
        "timestamp": [datetime(2026, 6, 8, 13, 31), datetime(2026, 6, 8, 13, 32)],
        "open": [100, 102],
        "high": [103, 105],
        "low": [99, 101],
        "close": [102, 104],
        "volume": [10, 20],
    }
)

agg = TimeframeAggregator()
timeframes = ["1min", "5min", "15min", "30min", "1h", "4h", "6h",
              "12h", "1d", "1w", "1m", "1q", "1y"]
# Show the first row for every target. The 1min result has two rows;
# every other target combines both input rows into one.
summary = pl.concat([
    agg.aggregate(bars, tf).head(1).with_columns(pl.lit(tf).alias("timeframe"))
    for tf in timeframes
]).select("timeframe", "timestamp", "open", "high", "low", "close", "volume")
with pl.Config(tbl_rows=20, tbl_width_chars=120):
    print(summary)
```

```text
shape: (13, 7)
┌───────────┬─────────────────────┬──────┬──────┬─────┬───────┬────────┐
│ timeframe ┆ timestamp           ┆ open ┆ high ┆ low ┆ close ┆ volume │
│ ---       ┆ ---                 ┆ ---  ┆ ---  ┆ --- ┆ ---   ┆ ---    │
│ str       ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64 ┆ i64   ┆ i64    │
╞═══════════╪═════════════════════╪══════╪══════╪═════╪═══════╪════════╡
│ 1min      ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 103  ┆ 99  ┆ 102   ┆ 10     │
│ 5min      ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 15min     ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 30min     ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 1h        ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 4h        ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 6h        ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 12h       ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 1d        ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 1w        ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 1m        ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 1q        ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
│ 1y        ┆ 2026-06-08 13:31:00 ┆ 100  ┆ 105  ┆ 99  ┆ 104   ┆ 30     │
└───────────┴─────────────────────┴──────┴──────┴─────┴───────┴────────┘
```

An aggregate's timestamp is the **first observed input timestamp**, not the bucket boundary. Missing minutes are not filled; partial buckets remain. Aggregate one symbol at a time, since symbols are not grouped separately.

### Classification and history

```python
from datetime import datetime, timedelta

import polars as pl
from thestrat import classify_bars_df

bars = pl.DataFrame({
    "timestamp": [datetime(2026, 6, 8, 13, 30) + timedelta(minutes=i) for i in range(4)],
    "open": [100, 101, 102, 103],
    "high": [104, 103, 105, 106],
    "low": [98, 99, 100, 97],
    "close": [102, 102, 104, 98],
})
classified = classify_bars_df(bars)
view = classified.select("scenario0", "scenario1", "scenario2", "scenario3", "color0", "shape", "in_force")
with pl.Config(tbl_width_chars=120):
    print(view)
```

```text
shape: (4, 7)
┌───────────┬───────────┬───────────┬───────────┬────────┬───────┬──────────┐
│ scenario0 ┆ scenario1 ┆ scenario2 ┆ scenario3 ┆ color0 ┆ shape ┆ in_force │
│ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---    ┆ ---   ┆ ---      │
│ str       ┆ str       ┆ str       ┆ str       ┆ str    ┆ str   ┆ bool     │
╞═══════════╪═══════════╪═══════════╪═══════════╪════════╪═══════╪══════════╡
│ null      ┆ null      ┆ null      ┆ null      ┆ green  ┆ null  ┆ null     │
│ 1         ┆ null      ┆ null      ┆ null      ┆ green  ┆ null  ┆ false    │
│ 2U        ┆ 1         ┆ null      ┆ null      ┆ green  ┆ null  ┆ true     │
│ 3         ┆ 2U        ┆ 1         ┆ null      ┆ red    ┆ null  ┆ true     │
└───────────┴───────────┴───────────┴───────────┴────────┴───────┴──────────┘
```

Rows must already be in chronological order for one symbol and timeframe. `scenario0` and `color0` describe the current row; suffixes 1–3 describe prior rows. A first bar has no scenario or in-force value. The fourth row above has no `scenario3` because the first bar had no prior range.

### Session bucket boundaries

```python
from datetime import UTC, datetime

import polars as pl
from thestrat import InstrumentType, TimeframeAggregator, bucket_key_expr, session_for

equity = session_for(InstrumentType.EQUITY_US)
bars = pl.DataFrame({
    "timestamp": [datetime(2026, 6, 8, 13, 31, tzinfo=UTC), datetime(2026, 6, 8, 14, 29, tzinfo=UTC)],
    "open": [100, 102], "high": [103, 105], "low": [99, 101],
    "close": [102, 104], "volume": [10, 20],
})
keys = bars.select("timestamp", bucket_key_expr("1h", equity).alias("bucket"))
with pl.Config(tbl_width_chars=120):
    print(keys)
hourly = TimeframeAggregator().aggregate_for_instrument(bars, "1h", InstrumentType.EQUITY_US)
assert hourly.height == 1
assert hourly["timestamp"][0] == bars["timestamp"][0]
```

```text
shape: (2, 2)
┌─────────────────────────┬─────────────────────────┐
│ timestamp               ┆ bucket                  │
│ ---                     ┆ ---                     │
│ datetime[μs, UTC]       ┆ datetime[μs, UTC]       │
╞═════════════════════════╪═════════════════════════╡
│ 2026-06-08 13:31:00 UTC ┆ 2026-06-08 13:30:00 UTC │
│ 2026-06-08 14:29:00 UTC ┆ 2026-06-08 13:30:00 UTC │
└─────────────────────────┴─────────────────────────┘
```

Here 13:30 UTC is 09:30 New York time. Sessions align hour-and-above buckets; they do not filter trading hours or holidays. Naive aggregate timestamps are assumed UTC. Direct session-aware `bucket_key_expr` callers should supply timezone-aware timestamps. See [behavior](docs/behavior.md) for calendar and daylight-saving details.

### Multi-symbol and scalar helpers

```python
from datetime import datetime

import polars as pl
from thestrat import Color, Scenario, classify_bar, classify_bars_multi_symbol, classify_color, classify_scenario

prior = {"symbol": "ABC", "timestamp": datetime(2026, 6, 8), "timeframe": "1d",
         "open": 100, "high": 104, "low": 96, "close": 102, "volume": 1000}
current = {**prior, "timestamp": datetime(2026, 6, 9), "high": 105, "low": 95, "close": 103}
bar = classify_bar(current, prior)
assert bar["scenario0"] == Scenario.THREE
assert classify_color(100, 101) == Color.GREEN
assert classify_scenario(105, 95, 104, 96) == Scenario.THREE

# Separate history for each symbol; keep a single timeframe per call.
many = pl.DataFrame([prior, current, {**prior, "symbol": "XYZ"}]).sort("symbol", "timestamp")
classified_many = classify_bars_multi_symbol(many)
assert classified_many["scenario0"].to_list() == [None, "3", None]
```

See [API signatures and types](docs/api.md) for every root export and [calculation rules and limits](docs/behavior.md) for precise semantics. Financial disclaimers are in [DISCLAIMER.md](DISCLAIMER.md); code restrictions are in [LICENSE.md](LICENSE.md).
