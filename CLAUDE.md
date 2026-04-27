# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

`thestrat` is a small, portable Polars-based library that other repos consume for:

1. **Timeframe aggregation** — vectorized OHLCV resampling across all standard intra/inter-day timeframes.
2. **Strat bar classification** — scenario / color / shape / in-force, single-symbol and multi-symbol.

It has **no I/O, no async, no global state, no broker integrations.** Pure functions over Polars DataFrames.

## 1.0.0 Rewrite

Version 1.0 is a clean rewrite. Pre-1.0 modules (`aggregation.py`, `base.py`, `factory.py`, `indicators.py`, `precision.py`, `schemas.py`, `signals.py`) are now stubs that `raise ImportError` immediately. Any defunct project pinned to those import paths fails loudly when reinstalled — that is intentional and signals to the maintainer to delete or migrate the project.

**Do not restore the legacy modules.** If a feature from the legacy implementation is needed (signal metadata, swing-point indicators, factory pattern, etc.), build it cleanly inside the consuming project — not here. This package's design center is "small, portable primitives."

## File layout

```
thestrat/
  __init__.py        # Public API — re-exports from aggregator, classifier, types
  aggregator.py      # TimeframeAggregator
  classifier.py      # classify_bars_df, classify_bars_multi_symbol, classify_bar, classify_color, classify_scenario
  types.py           # BarDict, ClassifiedBar, Color, Scenario, Shape, Timeframe
  aggregation.py     # DEFUNCT stub
  base.py            # DEFUNCT stub
  factory.py         # DEFUNCT stub
  indicators.py      # DEFUNCT stub
  precision.py       # DEFUNCT stub
  schemas.py         # DEFUNCT stub
  signals.py         # DEFUNCT stub
tests/
  test_aggregator.py
  test_aggregation_accuracy.py
  test_classifier.py
```

## Development

### Setup
```bash
uv sync --extra test --extra dev
```

### Testing
```bash
uv run pytest                                        # all tests
uv run pytest tests/test_classifier.py -v            # one file
uv run pytest --cov=thestrat --cov-report=term-missing
```

### Code quality
```bash
uv run ruff check .
uv run ruff format .
uv run pyright
```

## Design rules

- **Pure functions only.** Aggregator and classifier do not log, raise on missing data, or mutate state. They take a Polars DataFrame and return a Polars DataFrame.
- **No new dependencies without a concrete reason.** The runtime dep set is `polars[timezone]`. That is the correct number; resist adding pandas / numpy / pydantic / pytz unless a real consumer can't be served otherwise.
- **No backwards-compatibility shims.** Per the original Zero-Technical-Debt policy, replacements are clean cuts. The legacy stubs exist solely to surface a loud error to defunct consumers; they are not a migration path.
- **Tests live alongside the public API.** When adding a function, add tests in the same test file as its closest siblings. Don't introduce a new test module unless adding a meaningfully separate API.

## Consumers

Known internal consumers (April 2026):
- `StratTraderMCP` — currently still uses its own local copy of the aggregator and classifier, marked deprecated. Migration tracked separately.
- `StratBacktester` — Phase 1 futures backtester. Will start consuming `thestrat` for the bar-stitcher slice.
- `StratStrategyResearch` — knowledge-corpus repo; no code import yet, may consume later.

Changes here ripple outward. Any breaking change to the public API requires updating all known consumers in the same change wave.
