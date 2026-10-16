# TheStrat

[![Tests](https://github.com/jlixfeld/thestrat/actions/workflows/tests.yml/badge.svg)](https://github.com/jlixfeld/thestrat/actions/workflows/tests.yml)
[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/jlixfeld/thestrat)
[![License](https://img.shields.io/badge/license-Private-red)](https://github.com/jlixfeld/thestrat)

A Polars library for session-aware OHLCV aggregation and Strat bar classification through one `TheStrat` instance.

The 2.0 API is a hard cut. Use the interface documented here; previous free classifiers, aggregator classes and legacy modules are removed without aliases or compatibility wrappers.

The source code is all rights reserved. Read the [license, community and support notices](#notices).

## Contents

- [Documentation](#documentation)
- [Installation](#installation)
- [Aggregation](#aggregation)
- [Classification](#classification)
- [Snapshot updates](#snapshot-updates)
- [Mixed source aggregation](#mixed-source-aggregation)
- [Scalar classification](#scalar-classification)
- [Window geometry](#window-geometry)
- [State and ownership](#state-and-ownership)
- [Sessions](#sessions)
- [Extension](#extension)
- [Development verification](#development-verification)
- [Notices](#notices)

## Documentation

- [API reference](docs/api.md): Signatures and schemas.
- [Calculation and state behavior](docs/behavior.md): Calculations and caller responsibilities.
- [Agent documentation](llms.txt): An index of references and examples for agents.

## Installation

```sh
uv add git+https://github.com/jlixfeld/thestrat.git
```

For a local checkout, use `uv sync --extra test --extra dev`.

## Usage walkthrough

### Aggregation

Each step below includes its actual result. Omitting `symbol` is also supported and creates one unnamed series with null symbols.

#### Supply source minutes

These five complete one-minute snapshots identify interval starts. Every timestamp in this walkthrough is UTC. Run the Python blocks in order in one session.

```python
from datetime import datetime, timedelta, timezone

import polars as pl

from thestrat import InstrumentType, TheStrat, session_for

start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)  # 09:30 Eastern
minutes = pl.DataFrame({
    "symbol": ["EXAMPLE"] * 5,
    "timeframe": ["1min"] * 5,
    "timestamp": [start + timedelta(minutes=i) for i in range(5)],
    "open": [100.0, 101.0, 102.0, 101.0, 103.0],
    "high": [102.0, 103.0, 103.0, 104.0, 105.0],
    "low": [99.0, 100.0, 100.0, 101.0, 102.0],
    "close": [101.0, 102.0, 101.0, 103.0, 104.0],
    "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
})
input_preview = minutes.select("symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume")
```

<!-- output: input_preview -->

| symbol | timeframe | timestamp (UTC) | open | high | low | close | volume |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXAMPLE | 1min | 2026-01-05 14:30 | 100.0 | 102.0 | 99.0 | 101.0 | 10.0 |
| EXAMPLE | 1min | 2026-01-05 14:31 | 101.0 | 103.0 | 100.0 | 102.0 | 20.0 |
| EXAMPLE | 1min | 2026-01-05 14:32 | 102.0 | 103.0 | 100.0 | 101.0 | 30.0 |
| EXAMPLE | 1min | 2026-01-05 14:33 | 101.0 | 104.0 | 101.0 | 103.0 | 40.0 |
| EXAMPLE | 1min | 2026-01-05 14:34 | 103.0 | 105.0 | 102.0 | 104.0 | 50.0 |

<!-- /output -->

#### Aggregate explicitly

Creating the instance stores the supplied frame. Calling `aggregate()` calculates bars for the configured timeframes. The result has seventeen rows: five source minutes and twelve higher-timeframe rows. This preview shows the latest row for each of the thirteen timeframes; the `1min` row is the latest source minute.

```python
strat = TheStrat(session=session_for(InstrumentType.EQUITY_US), bars=minutes)
aggregated = strat.aggregate()
aggregate_preview = aggregated.group_by("timeframe", maintain_order=True).last().select(
    "symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume",
)
```

<!-- output: aggregate_preview -->

| symbol | timeframe | timestamp (UTC) | open | high | low | close | volume |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXAMPLE | 1min | 2026-01-05 14:34 | 103.0 | 105.0 | 102.0 | 104.0 | 50.0 |
| EXAMPLE | 5min | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 15min | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 30min | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 1h | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 4h | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 6h | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 12h | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 1d | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 1w | 2026-01-05 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 1m | 2026-01-01 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 1q | 2026-01-01 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |
| EXAMPLE | 1y | 2026-01-01 14:30 | 100.0 | 105.0 | 99.0 | 104.0 | 150.0 |

<!-- /output -->

Aggregation uses the first open, maximum high, minimum low, latest close and summed volume for each symbol/window. Output timestamps are interval starts, including when the opening minute is missing. `1min` means minute; `1m` means calendar month. Higher-timeframe rows can still be forming.

### Classification

#### Classify the calculated bars

Classification adds labels without changing `strat.bars`. Here the default fields include scenario, color, shape and in-force. This thirteen-timeframe preview selects current scenario and color alongside OHLCV; the following view selects recent classified rows with Polars.

```python
classified = strat.classify(aggregated)
preview = classified.group_by("timeframe", maintain_order=True).last().select(
    "timeframe", "timestamp", "open", "high", "low", "close", "volume",
    "scenario", "color",
)
```

<!-- output: preview -->

| timeframe | timestamp (UTC) | open | high | low | close | volume | scenario | color |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
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

<!-- /output -->

#### Inspect recent classified rows

Classify the supplied bars with their predecessor context first, then select recent rows. Truncating the input before classification would lose the first retained row's scenario and in-force context. This example shows the latest three source-minute rows for EXAMPLE; callers can also select recent rows per group with Polars `group_by("symbol", "timeframe").tail(3)`.

```python
recent_minutes = classified.filter(pl.col("timeframe") == "1min").select(
    "symbol", "timeframe", "timestamp", "scenario", "color", "shape", "in_force",
).tail(3)
```

<!-- output: recent_minutes -->

| symbol | timeframe | timestamp (UTC) | scenario | color | shape | in_force |
| --- | --- | --- | --- | --- | --- | --- |
| EXAMPLE | 1min | 2026-01-05 14:32 | 1 | red | null | false |
| EXAMPLE | 1min | 2026-01-05 14:33 | 2U | green | null | false |
| EXAMPLE | 1min | 2026-01-05 14:34 | 2U | green | null | false |

<!-- /output -->

### Snapshot updates

#### Prepare a revised snapshot

The replacement snapshot has the same minute key and complete OHLCV values. Its volume is now 70 instead of 50.

```python
revised = minutes.tail(1).with_columns(
    pl.lit(106.0).alias("high"),
    pl.lit(105.0).alias("close"),
    pl.lit(70.0).alias("volume"),
)
revised_preview = revised.select("symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume")
```

<!-- output: revised_preview -->

| symbol | timeframe | timestamp (UTC) | open | high | low | close | volume |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXAMPLE | 1min | 2026-01-05 14:34 | 103.0 | 106.0 | 102.0 | 105.0 | 70.0 |

<!-- /output -->

#### Recalculate from the replacement

The returned frame still contains bars for every configured timeframe. This projection shows the corrected source minute and its four-hour aggregate. Higher-timeframe volume changes from 150 to 170 because the snapshot replaces the old contribution.

```python
updated = strat.aggregate(revised, mode="snapshot")
updated_preview = updated.filter(pl.col("timeframe").is_in(["1min", "4h"])).filter(
    (pl.col("timeframe") == "4h") | (pl.col("timestamp") == revised[0, "timestamp"])
).select("symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume")
```

<!-- output: updated_preview -->

| symbol | timeframe | timestamp (UTC) | open | high | low | close | volume |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXAMPLE | 1min | 2026-01-05 14:34 | 103.0 | 106.0 | 102.0 | 105.0 | 70.0 |
| EXAMPLE | 4h | 2026-01-05 14:30 | 100.0 | 106.0 | 99.0 | 105.0 | 170.0 |

<!-- /output -->

#### Refresh classification explicitly

Aggregation does not refresh labels automatically. Classify the updated frame, then inspect all four current fields for those same two rows. The revised minute closes above its prior high of 104, so in-force becomes true; the four-hour bar still has no predecessor.

```python
updated_labels = strat.classify(updated)
updated_label_preview = updated_labels.filter(pl.col("timeframe").is_in(["1min", "4h"])).filter(
    (pl.col("timeframe") == "4h") | (pl.col("timestamp") == revised[0, "timestamp"])
).select("symbol", "timeframe", "timestamp", "scenario", "color", "shape", "in_force")
```

<!-- output: updated_label_preview -->

| symbol | timeframe | timestamp (UTC) | scenario | color | shape | in_force |
| --- | --- | --- | --- | --- | --- | --- |
| EXAMPLE | 1min | 2026-01-05 14:34 | 2U | green | null | true |
| EXAMPLE | 4h | 2026-01-05 14:30 | null | green | null | null |

<!-- /output -->

### Mixed source aggregation

Supply closed higher-timeframe bars alongside complete minute snapshots when the full source-minute history is not retained. `SourceCoverage` states which source intervals are complete, including no-trade portions, and separately lists every expected source key. Dates in this example are UTC; the closed five-minute bar covers 09:30–09:35 Eastern.

```python
from thestrat import SourceCoverage

closed = aggregated.filter(pl.col("timeframe") == "5min")
later = pl.DataFrame({
    "symbol": ["EXAMPLE", "EXAMPLE"],
    "timeframe": ["1min", "1min"],
    "timestamp": [start + timedelta(minutes=5), start + timedelta(minutes=6)],
    "open": [104.0, 105.0], "high": [106.0, 107.0],
    "low": [103.0, 104.0], "close": [105.0, 106.0], "volume": [60.0, 70.0],
}).select(closed.columns)
mixed_sources = pl.concat([closed, later])
coverage = SourceCoverage(
    intervals=pl.DataFrame({
        "symbol": ["EXAMPLE", "EXAMPLE"], "timeframe": ["5min", "1min"],
        "start": [start, start + timedelta(minutes=5)],
        "end": [start + timedelta(minutes=5), start + timedelta(minutes=7)],
    }),
    expected_keys=mixed_sources.select("symbol", "timeframe", "timestamp"),
)
mixed = TheStrat(session=session_for("equity_us"), timeframes=["1h", "5min"], bars=mixed_sources)
mixed_result = mixed.aggregate(coverage=coverage, through=start + timedelta(minutes=6))
mixed_preview = mixed_result.select("timeframe", "timestamp", "open", "high", "low", "close", "volume")
```

<!-- output: mixed_preview -->

| timeframe | timestamp (UTC) | open | high | low | close | volume |
| --- | --- | --- | --- | --- | --- | --- |
| 1h | 2026-01-05 14:30 | 100.0 | 107.0 | 99.0 | 106.0 | 280.0 |
| 5min | 2026-01-05 14:35 | 104.0 | 107.0 | 103.0 | 106.0 | 130.0 |

<!-- /output -->

`mixed.bars` still contains the three supplied source rows. The calculated rows are returned separately, so they do not silently become additional source contributions. Each configured timeframe receives only its current window at `through`, an inclusive source-minute start. Coverage ends are exclusive. An expected source row that disappears raises an error; an explicitly covered no-trade interval remains empty. New snapshot keys must already be listed in the supplied manifest. Retain that manifest independently when filtering source data.

Closed source windows must fit the required eligible segments. A final higher-timeframe bar extending into the future is rejected. A correction overlapping a retained summary requires a coherent replacement source set and declaration. Callers own retrieval, retention, rollover and archive repairs; the library selects compatible disjoint contributions using the configured session calendar. See [coverage and limits](docs/api.md#mixed-source-coverage).

#### Classify without volume

Classification uses OHLC, grouping and timestamps. It does not require or fabricate volume; supplied volume is preserved as unused metadata. Aggregation still requires numeric OHLCV.

```python
price_only = mixed.classify(mixed_result.drop("volume"), fields=["color"]).select(
    "timeframe", "open", "close", "color",
)
```

<!-- output: price_only -->

| timeframe | open | close | color |
| --- | --- | --- | --- |
| 1h | 100.0 | 106.0 | green |
| 5min | 104.0 | 106.0 | green |

<!-- /output -->

### Scalar classification

#### Classify a scalar

A scalar call performs only the selected current-bar calculation. `Color.GREEN` has the string value shown below.

```python
color = strat.classify.scalar(kind="color", open_price=100.0, close_price=101.0)
```

<!-- output: color -->

```text
green
```

<!-- /output -->

### Window geometry

#### Inspect containing and neighboring windows

`window()` accepts one aware timestamp or a sequence. This batch asks for one preceding and one following window per timestamp/timeframe. Position zero contains the input in an eligible segment; positions -1 and +1 are its neighbors. Input order and repeated inputs are preserved. All geometry timestamps and segment boundaries below are UTC.

```python
windows = strat.window(
    [start, start + timedelta(minutes=90)],
    timeframes=["1h", "4h"], before=1, after=1,
)
```

<!-- output: windows -->

| input_timestamp (UTC) | timeframe | position | window_start (UTC) | window_end (UTC) | segments |
| --- | --- | --- | --- | --- | --- |
| 2026-01-05 14:30 | 1h | -1 | 2026-01-02 20:30 | 2026-01-02 21:00 | [2026-01-02 20:30, 2026-01-02 21:00) |
| 2026-01-05 14:30 | 1h | 0 | 2026-01-05 14:30 | 2026-01-05 15:30 | [2026-01-05 14:30, 2026-01-05 15:30) |
| 2026-01-05 14:30 | 1h | 1 | 2026-01-05 15:30 | 2026-01-05 16:30 | [2026-01-05 15:30, 2026-01-05 16:30) |
| 2026-01-05 14:30 | 4h | -1 | 2026-01-02 18:30 | 2026-01-02 21:00 | [2026-01-02 18:30, 2026-01-02 21:00) |
| 2026-01-05 14:30 | 4h | 0 | 2026-01-05 14:30 | 2026-01-05 18:30 | [2026-01-05 14:30, 2026-01-05 18:30) |
| 2026-01-05 14:30 | 4h | 1 | 2026-01-05 18:30 | 2026-01-05 21:00 | [2026-01-05 18:30, 2026-01-05 21:00) |
| 2026-01-05 16:00 | 1h | -1 | 2026-01-05 14:30 | 2026-01-05 15:30 | [2026-01-05 14:30, 2026-01-05 15:30) |
| 2026-01-05 16:00 | 1h | 0 | 2026-01-05 15:30 | 2026-01-05 16:30 | [2026-01-05 15:30, 2026-01-05 16:30) |
| 2026-01-05 16:00 | 1h | 1 | 2026-01-05 16:30 | 2026-01-05 17:30 | [2026-01-05 16:30, 2026-01-05 17:30) |
| 2026-01-05 16:00 | 4h | -1 | 2026-01-02 18:30 | 2026-01-02 21:00 | [2026-01-02 18:30, 2026-01-02 21:00) |
| 2026-01-05 16:00 | 4h | 0 | 2026-01-05 14:30 | 2026-01-05 18:30 | [2026-01-05 14:30, 2026-01-05 18:30) |
| 2026-01-05 16:00 | 4h | 1 | 2026-01-05 18:30 | 2026-01-05 21:00 | [2026-01-05 18:30, 2026-01-05 21:00) |

<!-- /output -->

`window_start` is the canonical aggregation label; `window_end` is the last eligible segment end, including an early close. The `segments` list gives the actual eligible half-open intervals. A timestamp outside those intervals has no position-zero row. Neighbor counts default to zero; omit `timeframes` to use configured timeframes. A missing requested neighbor raises `ScheduleRangeError` rather than returning a shorter result. The method schedules no work and reads no clock.

## State and ownership

`strat.bars` initially stores the supplied frame. After ordinary `aggregate()`, it contains source minutes and calculated timeframe bars. With `coverage=...`, it contains only supplied source rows after snapshots; current calculated rows are returned separately. Callers can filter it with Polars and assign the result back. Assignment alone does no calculation. `aggregate` applies snapshots and calculates bars; `classify` returns labels without aggregating or changing the instance's frame.

Callers normalize provider fields, units and timestamps, sanitize duplicates, and choose retention. Ordinary minute aggregation requires underlying source minutes to update a retained aggregate-only window. Mixed aggregation requires an explicitly complete compatible source cover. Classification of a summary itself is valid. Do not prune those source minutes and expect later corrections to reconstruct them.

Only snapshot updates are implemented. Passing `mode="delta"` raises `NotImplementedError` before state changes. Invalid inputs still raise clear validation errors.

## Sessions

Equity presets use the exchange calendar, including holidays and early closes. Session settings choose the intraday anchor once; higher intraday windows and window lookup share it. Futures use the configured exchange session across midnight; crypto uses continuous UTC windows. Custom schedules can supply other sessions and breaks.

## Extension

Classification fields are scenario, color, shape and in-force. Scalar kinds additionally include a whole-bar result. Future classification types can extend these components with explicit inputs, dependencies, outputs, predecessor context and tests; signal detection and order handling are outside this API.

## Development verification

```sh
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

The tests execute the README example and compare its displayed rows with the calculated results.

## Notices

### Important Notice About Terminology and Community

**"The Strat"** refers to the trading methodology developed by the late Rob Smith and fostered by the broader trading community. I make **NO CLAIMS** to ownership of:

- The Strat methodology or concepts
- The terminology "The Strat"
- Any trading strategies or techniques discussed in the community
- The broader educational content around these concepts

This repository contains **MY IMPLEMENTATION** of software tools related to The Strat concepts. The community and methodology existed long before this code and will continue regardless of this repository.

### Code License: No License (All Rights Reserved)

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

### Why This Distinction Matters

I'm sharing my code to contribute to The Strat community, but I'm not providing free tech support for my implementation. If you want to build Strat tools:

1. **Study the concepts** (freely available from Rob Smith and the community)
2. **Build your own implementation** (which you're completely free to do)
3. **Don't copy my code** (but feel free to be inspired by the approach)

### No Support Policy

I don't provide support for this code. The Strat community has many great resources for learning the methodology itself, but I won't be answering questions about my specific implementation.
