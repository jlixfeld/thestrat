# Mixed-timeframe prototype

For the current public API, see [supported-path measurements](supported_mixed_report.md).

Historical prototype evidence: these measurements predate the supported `TheStrat.aggregate(coverage=..., through=...)` path. The public calculation is documented in `../docs/api.md`; these files remain experimental evidence and are not runtime dependencies.

The prototype reconstructs current OHLCV from a small complete mix of minute and higher-timeframe bars. It is isolated under `experiments/`; the public `TheStrat` API was unchanged by that historical prototype.

The result supports bounded working state, but **does not establish a single-symbol speed advantage over a focused full-minute calculation**. In this run, warm mixed calculation took 29.3 ms versus 24.2 ms for the focused comparator. Updating one new minute took 27.8 ms for mixed state versus 30.1 ms for the focused calculation; these small differences should not be treated as a capacity guarantee.

## What was measured

Deterministic default NYSE data for 2026, including the calendar's holidays, early closes and DST. The initial evaluation is December 30 at 15:05 Eastern: 97,086 source minutes per symbol, out of 97,530 generated year minutes. All inputs use one consistent price basis and volume unit. Symbols have identical deterministic values but distinct keys in candidate batches.

The public-library comparator computes 129,522 historical rows per symbol and then selects the 13 current rows. The focused comparator scans the same source minutes but emits only current rows. Both comparators execute separate calls per logical symbol, reusing the normalized single-symbol fixture to avoid allocating 100 years of minutes simultaneously. The candidate handles all symbols in one Polars batch. Multi-symbol ratios therefore combine a batching advantage with source reduction; they are **not** an equal-batching comparison.

Wall milliseconds per entire batch, median unless marked single sample:

| Operation | 1 symbol | 10 symbols | 100 symbols | Samples per count |
| --- | ---: | ---: | ---: | --- |
| Mixed current calculation, no incoming snapshot | 29.3 | 35.3 | 59.1 | 25 |
| Mixed repeated-minute snapshot replacement | 31.6 | 38.6 | 48.2 | 25 |
| Mixed genuinely advancing new minute | 27.8 | 37.7 | 61.2 | 20 |
| Focused current bars from all minutes | 24.2 | 291.0 | 2,433.6 | 5 / 3 / 1 |
| Public-library full-history rebuild and current selection | 636.5 | 6,383.9 | 61,305.4 | 5 / 3 / 1 |
| Older-hour restoration and current ancestors | 268.2 | 293.3 | 383.9 | 1 |
| Hour rollover hydration | 117.2 | 153.5 | 191.1 | 1 |
| Day rollover hydration | 145.9 | 147.5 | 199.0 | 1 |

The 100-symbol baseline numbers are actual timed computations, not extrapolations. They have no meaningful percentile estimate. The JSON includes sample counts, ranges, p95 where at least 20 samples exist, and process CPU time. The p95 estimator is the observed upper order statistic `sorted[min(n-1, floor(n*0.95))]`; at n=20 this is the maximum.

New-minute samples advance from 15:06 through 15:25, each on the preceding state. Candidate timings include normalization, snapshot replacement, coverage checks, cached geometry selection and invalidation, and reduction. Source fixture selection and correctness comparisons are outside timers. Focused advancing one-symbol calculation, measured separately, includes geometry selection and reduction but excludes source ingestion/validation. Neither current-computation path refetches calendar geometry for each minute after hour preparation.

## Preparation, restoration and memory

For one symbol, generation took 88.4 ms, creating the minute/HTF archive 262.3 ms, initial canonical geometry 141.3 ms, and seed selection 6.9 ms. Cold costs are separate from steady update timings. Classification, predecessor context, network feeds, databases and trading execution are excluded.

Initial mixed state has 66 rows per symbol: 36 minutes, 5 closed hours, 20 daily bars, 2 months and 3 quarters. State and geometry are visible; no hidden source-year cache exists in the candidate. The fixture and oracle retain source history separately.

| Visible allocation estimate | 1 symbol | 100 symbols |
| --- | ---: | ---: |
| Mixed bar DataFrame | 3,504 bytes | 356,340 bytes |
| Required-key coverage manifest | 864 bytes | 92,340 bytes |
| Active geometry and current-hour short windows | 153,521 bytes shared | 153,521 bytes shared |

The single-symbol source DataFrame is 5,242,644 estimated bytes. These are Polars data-buffer estimates and a recursive Python geometry estimate, not isolated peak RSS. Whole-process high-water RSS was 621,232,128 bytes, including fixtures, library outputs, allocator retention, validation and every workload. No candidate-only peak-memory claim is made.

Rollover is an explicit bounded reseed. The caller supplies already-computed closed HTF summaries and new-hour minutes; the timed path prepares fresh canonical geometry, selects the seed and calculates. Hour rollover reads 32 rows per symbol (1 minute); day rollover reads 27 (1 minute). At 100 symbols these inputs occupy 169,480 and 143,030 estimated bytes respectively. The harness separately records its one-symbol archive rebuilding cost, about 201–260 ms, rather than hiding it inside a claimed free source. A production caller would need to maintain or retrieve those closed summaries. The prototype does not implement automatic sealing or archive maintenance.

Older-hour repair receives 60 complete source minutes per symbol (6,000 rows / 329,400 estimated bytes for 100 symbols), rebuilds that hour and recalculates current ancestors. Retrieval/filtering of those supplied minutes is outside the restoration timer. This is simulated local restoration, not a database latency measurement.

## Correctness and limits

Children are disjoint within each reduction: minutes feed subhour/hour, hours feed intraday/day, days feed week/month, months feed quarter, quarters feed year. Daily retention covers the union of current month and week, including prior-month or prior-year days. Weekly bars never feed months.

The caller explicitly declares point-in-time completeness, its starting coverage bound, and watermark. The starting bound must cover the current year plus any previous-year portion of the current week. A required-key manifest detects deleted constituent rows. These declarations are provenance requirements, not proof obtainable from trade density: no-trade minutes and entirely empty current subhour windows are valid and produce no fabricated rows. Declaring incomplete source history complete can still produce wrong results.

Snapshots replace source keys, so 203→296 changes aggregate volume by 93. Extrema can retract; correcting an early open does not overwrite a later close. Inputs outside the retained hour fail until their partition is restored. A missing manifest child also fails. Restore must cover exactly the named partition and symbol batch. If a restored partition overlaps another retained ancestor or descendant, the prototype rejects it atomically and requires a fresh consistent seed. It does not silently repair every overlapping archive level.

Outputs are current windows at the original watermark, plus the explicitly restored partition returned by `restore`. The prototype does not emit or repair every affected closed historical window, support deletion-to-empty restoration, build a general composition API, or implement Backtester migration. It is restricted to the default NYSE anchor; other session/anchor arrangements need their own composition validation.

Twelve prototype tests cover independent OHLCV arithmetic, repeat snapshots, retractions, valid sparse/no-trade intervals, dropped children, advancing minutes, full-minute comparison across 13 timeframes, March/April week/month/quarter overlap, early close/DST, January prior-year coverage, year rollover, older-hour repair and both directions of overlapping-summary rejection. Three restoration regressions were reproduced before fixing them during independent review.

The first pilot is preserved in `mixed_initial_pilot.json`: one-symbol snapshot/new-minute medians were 118.4/211.7 ms. That implementation redundantly queried calendar geometry. Reusing the existing canonical short-window generator and current-hour eligibility removed that measured overhead; the final run is in `mixed_metrics.json`.

## Reproduce

From the repository root, with dependencies already installed:

```sh
UV_CACHE_DIR=/tmp/thestrat-uv uv run --no-sync python -m experiments.benchmark_mixed --output experiments/mixed_metrics.json
UV_CACHE_DIR=/tmp/thestrat-uv uv run --no-sync python -m pytest
UV_CACHE_DIR=/tmp/thestrat-uv uv run --no-sync ruff check .
UV_CACHE_DIR=/tmp/thestrat-uv uv run --no-sync ruff format --check .
UV_CACHE_DIR=/tmp/thestrat-uv uv run --no-sync pyright
UV_CACHE_DIR=/tmp/thestrat-uv uv run --no-sync pyright experiments tests/test_mixed_prototype.py
```

The existing environment was reused because sandboxed dependency resolution could not reach PyPI. Measurements used Python 3.14.3, Polars 1.40.1, macOS arm64 and an 18-thread Polars pool. This was a desktop microbenchmark with no isolation from other host activity. Keep the memory-saving design as an experiment; compare against an equally batched focused-minute implementation before deciding on a production migration.
