# Supported mixed aggregation measurements

This measures `TheStrat.aggregate(coverage=..., through=...)`, including normalization, manifest and coverage validation, canonical source planning, geometry lookup, reduction and atomic source assignment. The smaller source set is useful, but these measurements **do not establish a speed advantage over an equally batched focused minute calculation**. This supported validation path is slower than the historical prototype, particularly at 100 symbols.

Run:

```sh
uv run python -m experiments.benchmark_supported_mixed --symbols 1,10,100 --output experiments/supported_mixed_metrics.json
```

[Raw measurements](supported_mixed_metrics.json) record wall time, process CPU, sample counts and ranges. Environment: macOS arm64, Python 3.14.3, Polars 1.40.1, 18 Polars threads. This was a shared desktop; test/static-check processes overlapped the initial portion of the run. Timing variation is uncontrolled, and these are neither isolated hardware measurements nor production capacity estimates. Historical prototype measurements remain unchanged in the other experiment files.

## Current outputs and comparisons

Synthetic homogeneous NYSE data has 97,086 eligible source minutes per symbol at December 30, 2026, 15:05 Eastern. Mixed input has 66 rows per symbol: recent minutes and closed hourly/daily/monthly/quarterly bars. Every candidate sample is checked for exact equality against all 13 current full-minute outputs outside the timed section. Prices and volumes use exactly representable synthetic values.

Wall milliseconds per entire batch, medians except single-sample entries:

| Operation | 1 symbol | 10 symbols | 100 symbols | Samples |
| --- | ---: | ---: | ---: | --- |
| Cold public mixed calculation | 178.7 | 216.5 | 883.4 | 1 each |
| Warm mixed calculation | 35.0 | 67.8 | 343.0 | 10 each |
| Minute snapshot replacement | 39.6 | 71.2 | 485.2 | 20 each |
| Strictly advancing new minute | 43.3 | 81.1 | 505.2 | 20 each |
| Focused current-minute, equal batch | 31.3 | 66.6 | not measured | 3 each |
| Focused current-minute, streamed | 58.1 | 294.7 | 2,183.8 | 3 / 1 / 1 |
| Ordinary public full-minute, streamed | 498.4 | 5,695.5 | 46,942.5 | 3 / 1 / 1 |

The focused comparator uses already eligible minute rows and precomputed public window geometry; it performs no source normalization or completeness validation. Its separate cold geometry preparation took 132.2 ms. Mixed warm timings include their own cached geometry lookup and source validation. Neither warm path includes caller retrieval, input construction or price ingestion. New-minute samples genuinely advance 20 timestamps; the caller builds each extended interval declaration and final manifest before timing. Snapshot samples alternate full replacement volumes, never additive deltas. Correctness checks and fixture resets are outside timers.

The ordinary comparator produces historical aggregates before selecting current outputs. Streamed baselines execute one real call per logical symbol using a normalized shared fixture; candidate calls process a single multi-symbol batch. Their ratios combine different output work and batching and are not direct algorithm speedups. The equal-batch focused comparison is included for 1 and 10 symbols; 100 full-year copies were deliberately not allocated. Single-sample baselines have no meaningful percentile estimate.

Process CPU also gives no universal improvement: one-symbol snapshot/new-minute medians were 135.0/124.5 ms versus 43.6 ms for equal-batch focused calculation. At 10 symbols they were 223.6/207.6 ms versus 367.8 ms focused. Multi-threaded process CPU can exceed wall time. The complete raw ranges should be used when assessing these noisy samples.

## Preparation, rollover and retained state

One-symbol fixture generation took 110.2 ms; computing its closed-summary archive took 166.4 ms; simulated retrieval and canonical retrieval bounds took 135.7 ms. This archive is caller-owned input, not free hidden library computation. Replicating identical values under distinct symbol keys is fixture setup outside timing, not a real 100-symbol archive build.

| Cold replacement seed | 1 symbol | 10 symbols | 100 symbols | Rows per symbol / minute rows |
| --- | ---: | ---: | ---: | --- |
| Next hour public calculation | 182.4 ms | 193.9 ms | 863.8 ms | 32 / 1 |
| Next day public calculation | 161.7 ms | 195.1 ms | 605.6 ms | 27 / 1 |

Each reload has one sample and includes cold internal geometry/planning. Separately measured one-symbol archive construction at reload was 153.8–194.2 ms and simulated retrieval was 136.7–155.6 ms. This harness rebuilds the caller archive from the year fixture to prepare each replacement seed; it is explicitly outside the public hydration measurement and is not an incremental retention implementation. No database, broker or network is involved. The library receives only the bounded rows shown above, never the full-year minutes for mixed calculation.

Initial Polars source/coverage/manifest buffers were 3,504 + 102 + 864 bytes for one symbol and 356,340 + 10,650 + 92,340 bytes for 100 symbols. These are frame estimates only, excluding internal geometry, Python objects, allocator overhead and peak RSS. They are not total-memory or peak-memory claims. Geometry cache entry counts are bounded; caller source retention is not library-managed.

The supported API does not retrieve sources, seal archives, repair historical outputs or coordinate late corrections. Callers must supply coherent declared inputs, and corrections overlapping retained summaries fail atomically. Broader session/anchor, sparse-data, DST/break and failure behavior is verified by tests, not extrapolated from this NYSE performance fixture. This is not evidence of a Backtester migration or a deployed consumer change.
