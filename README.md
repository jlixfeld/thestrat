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

A Polars library for session-aware OHLCV aggregation and Strat bar classification through one `TheStrat` instance.

The 2.0 API is a hard cut. Use the interface documented here; previous free classifiers, aggregator classes and legacy modules are removed without aliases or compatibility wrappers.

## Installation

```sh
uv add git+https://github.com/jlixfeld/thestrat.git
```

For a local checkout, use `uv sync --extra test --extra dev`. [API reference](docs/api.md) defines signatures and schemas; [behavior](docs/behavior.md) defines calculations and caller responsibilities.

## Aggregate and classify

This complete example supplies five normalized one-minute snapshots. Timestamps identify the start of each source minute, not message receipt time. Multiple symbols work through the same operations; omitting `symbol` creates one unnamed series with null symbols.

```python
from datetime import datetime, timedelta, timezone

import polars as pl

from thestrat import InstrumentType, TheStrat, session_for

start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)  # 09:30 Eastern
minutes = pl.DataFrame({
    "symbol": ["EXAMPLE"] * 5,
    "timestamp": [start + timedelta(minutes=i) for i in range(5)],
    "open": [100.0, 101.0, 102.0, 101.0, 103.0],
    "high": [102.0, 103.0, 103.0, 104.0, 105.0],
    "low": [99.0, 100.0, 100.0, 101.0, 102.0],
    "close": [101.0, 102.0, 101.0, 103.0, 104.0],
    "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
})

strat = TheStrat(session=session_for(InstrumentType.EQUITY_US), bars=minutes)
aggregated = strat.aggregate()
classified = strat.classify(aggregated, fields=["scenario", "color"], history=3)
preview = classified.group_by("timeframe", maintain_order=True).last().select(
    "timeframe", "timestamp", "open", "high", "low", "close", "volume",
    "scenario0", "color0",
)

# A revised snapshot replaces this minute's previous contribution.
revised = minutes.tail(1).with_columns(
    pl.lit(106.0).alias("high"),
    pl.lit(105.0).alias("close"),
    pl.lit(70.0).alias("volume"),
)
updated = strat.aggregate(revised, mode="snapshot")
updated_labels = strat.classify(updated)

# Scalars perform only the selected current-bar calculation.
color = strat.classify.scalar(kind="color", open_price=100.0, close_price=101.0)

# Scheduling stays with the caller; this only returns window starts.
windows = strat.window_starts(
    start, timeframes=["1h", "4h"], count=2, direction="forward",
)
```

One latest row per timeframe from `preview` is shown below. All timestamps are UTC. The first row shows the latest source minute; the other rows summarize the supplied minutes, including forming higher-timeframe bars. A null scenario means that no previous bar for that symbol/timeframe was supplied.

<!-- dataframe-preview: generated from the executable example -->

| timeframe | timestamp (UTC) | open | high | low | close | volume | scenario0 | color0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 12h | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 15min | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 1d | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 1h | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 1m | 2026-01-01 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 1min | 2026-01-05 14:34 | 103.0 | 105.0 | 102.0 | 104.0 | 50.0 | 2U | green |
| 1q | 2026-01-01 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 1w | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 1y | 2026-01-01 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 30min | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 4h | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 5min | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |
| 6h | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 | null | green |

Aggregation uses each symbol's first open, maximum high, minimum low, latest close and summed volume. It labels results with interval starts, including when the opening minute is missing. `1min` means minute; `1m` means calendar month. The default targets are all thirteen timeframes shown in the preview.

The revised minute changes the higher-timeframe volume from 150 to 170, not 220. Classifications run only when requested, and history stays separate per symbol and timeframe. `history=3` means offsets zero through three: current plus three prior values.

## State and ownership

`strat.bars` is the visible working DataFrame, containing source minutes and calculated targets. Callers can filter it with Polars and assign the result back. Assignment alone does no calculation. `aggregate` applies snapshots and calculates bars; `classify` returns labels without aggregating or changing the instance's frame.

Callers normalize provider fields, units and timestamps, sanitize duplicates, and choose retention. Updating an aggregate-only seed requires the source minutes behind that window; classification of the summary itself is valid. Do not prune those source minutes and expect later corrections to reconstruct them.

Only snapshot updates are implemented. Passing `mode="delta"` raises `NotImplementedError` before state changes. Invalid inputs still raise clear validation errors.

## Sessions and extension

Equity presets use the exchange calendar, including holidays and early closes. Session settings choose the intraday anchor once; higher intraday windows and window lookup share it. Futures use the configured exchange session across midnight; crypto uses continuous UTC windows. Custom schedules can supply other sessions and breaks.

Classification fields are scenario, color, shape and in-force. Scalar kinds additionally include a whole-bar result. Future classification types can extend these components with explicit inputs, dependencies, outputs, history needs and tests; signal detection and order handling are outside this API.

## Verification

```sh
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

The tests execute the README example and compare its displayed rows with the calculated results.
