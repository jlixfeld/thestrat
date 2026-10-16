# Calculation and state behavior

Read [api.md](api.md) for exact signatures and input schemas. All calculations are explicit. Assigning `strat.bars`, replacing a minute or pruning with Polars does not trigger work. Aggregation calculates OHLCV; classification is a separate call.

## Source snapshots and aggregation

Normalize provider messages to their source-minute interval start. A snapshot is the complete current OHLCV for that minute; it is not an additive volume contribution. Without coverage, the library replaces the stored source row by symbol/timeframe/timestamp, then calculates aggregated bars from source minutes only. Aggregate rows never feed back as source contributions.

For each symbol/window, timestamp order determines first open and last close; high is maximum, low minimum and volume summed. A corrected older minute cannot replace a later constituent's close. Replacing the earliest constituent can correct the open. Retractions of an extreme use the available underlying source set; they are not handled by retaining an obsolete maximum or minimum. A snapshot volume change from 203 to 296 changes an aggregate already containing 203 by 93. It does not add 296 again.

Both aggregation and classification sort their input; classification orders independently within each symbol and timeframe. Caller-sanitized inputs contain one row per key. A timestamp is interval identity, not receipt order. Multiple distinct snapshots for one source key across calls are legitimate replacements.

An aggregate-only summary can be classified but cannot be updated without the source minutes behind its window. The caller must retain or restore those minutes. Missing grid positions do not prove missing data: a minute without an eligible trade may legitimately be absent. No hidden history, automatic pruning or message deduplication is performed.

Aggregation stores the full source-plus-aggregate working frame in `bars`, and returns bars for selected timeframes. If `1min` is unselected, assigning the returned aggregate-only frame back to `bars` discards source history by the caller's choice. Caller metadata is not reduced into aggregate output. Classification preserves extra columns and does not mutate instance state; requested classification fields are recalculated rather than trusted as current.

## Windows and sessions

Intervals are half-open: a source minute exactly at the next boundary belongs to the next window. Output timestamps are interval starts even if the first observed minute is missing. Use exchange-local calendar boundaries with UTC output; fixed UTC offsets cannot represent daylight saving time correctly.

For a normal US-equity session:

| Alignment | First 4h label and eligible coverage | Next 4h coverage |
| --- | --- | --- |
| Bottom of hour | 09:30; [09:30, 13:30) | [13:30, 16:00) |
| Top of hour | 09:00; eligible data [09:30, 13:00) | [13:00, 16:00) |

Hourly, 4h, 6h and 12h use that same session anchor, with shortened final windows at close. A twelve-hour equity bar can match the daily OHLCV while retaining a distinct timeframe. Sub-hour intervals use local clock multiples. Holiday, break and early-close eligibility comes from the configured schedule. No source price rows are synthesized during closed periods.

Futures sessions can begin on the prior civil date and continue across midnight. Keep the configured exchange anchor; UTC midnight and broker restarts do not reset windows. The preset covers CME equity-index sessions; supply a product-appropriate schedule for other contracts. Crypto uses a continuous UTC day, including weekends, without an equity-style shutdown.

Daily bars cover a session. Weeks start Monday, months on day one, quarters Jan/Apr/Jul/Oct and years January one using the configured local anchor. These calendar timeframes combine eligible data across sessions rather than splitting at each daily close. Period labels may precede the first eligible session. This convention is explicit; a trade-date calendar with a different week convention needs a different agreed contract.

Repeated fall-back local hours merge into their wall-clock bucket, whose label uses the earlier occurrence. One-minute source intervals are the exception: each preserves its physical UTC identity in both aggregate output and lookup, so a snapshot replaces the correct minute in either occurrence. Higher sub-hour timeframes can merge corresponding repeated local windows. A nonexistent spring-forward boundary shifts forward one local hour. Aggregation and window lookup share the boundary calculation, including session clipping and DST.

`window(timestamp_or_sequence, timeframes=..., before=0, after=0)` reports canonical geometry. Its `window_end` is the last eligible segment end; `segments` includes every eligible interval of the full window, not just intervals near the query. For example, a four-hour equity window on November 27, 2026 runs from 09:30 to the 13:00 Eastern early close (14:30–18:00 UTC).

Position zero means actual membership in an eligible half-open segment. A break, weekend or holiday has no zero row. With no containing window, previous windows end at/before the query and following windows begin strictly after it; a window spanning a break around the query is excluded from both. With membership, neighboring positions follow distinct canonical labels. This matters during fall-back: a second-occurrence query can belong to a window labeled with the first occurrence. One-minute source windows continue to use their physical timestamps.

Counts are per input/timeframe, closest neighbors are -1/+1, and output preserves input order and duplicate timestamps. Requested neighbors are complete or an unavailable-range error is raised. Window queries do not modify working source state. Window lookup does not combine prices. Explicit mixed aggregation can combine compatible closed summaries with recent minutes under the coverage contract below; ordinary minute aggregation still requires retained/reloaded minutes.

## Classification

Every predecessor comparison stays within `(symbol, timeframe)`. Null symbol is a valid anonymous group. There is no cross-symbol or cross-timeframe comparison. Missing predecessors yield null scenario and in-force; current color and shape need no prior bar.

| Calculation | Rule |
| --- | --- |
| Color | close > open: green; close < open: red; equal: neutral |
| Scenario | Strict high > reference high and low < reference low: both 3, high only 2U, low only 2D, neither 1 |
| Shape | With range r, hammer when min(open, close) >= low + 0.67r; otherwise shooter when max(open, close) <= low + 0.33r; otherwise null. Zero range is null. The implementation uses exactly `1 - 0.33` for the upper-zone threshold. |
| In-force | 1: false. 2U or green 3: close > prior high. 2D or red 3: close < prior low. Neutral 3: false. Missing prior: null. |

Taking out a level requires strict inequality; equality is not a breakout and is not in-force. Shape threshold equality qualifies. Bar classification is not trade entry/exit or signal detection.

Classification returns one `scenario`, `color`, `shape` and `in_force` value per bar when all fields are selected. It accepts no history parameter and generates no lag columns. Scalar whole-bar results use the same current-field names. Classify with predecessor context first, then use Polars filtering, `tail` or grouped `tail` to inspect recent rows. Truncating the input first loses the first retained bar's scenario/in-force context.

Fields are selective: scenario-only work does not calculate shape/color for unselected fields. In-force computes its scenario/color dependencies internally. Scalar formulas and DataFrame formulas must agree, comparing enum values to frame strings. Vectorization does not imply zero cost; benchmarks should separate update/aggregation, classification and future signal work.

## Caller responsibilities and failures

Callers provide normalized, consistently adjusted data, choose retention and sanitize duplicates. They own feed connections, clocks, point-in-time visibility, scheduling, source freshness, persistence and recovery. A backtest must supply only source bars available at its current evaluation time; final future bars do not become valid merely because they are in a DataFrame.

Input validation rejects malformed required values and configuration. A calculation failure leaves the working frame unchanged. Snapshot is the only implemented ingestion mode; delta requests fail explicitly before mutation. Classification warm-up nulls are valid results, while missing source information for an aggregate correction is an error.

## Mixed source calculation

Passing `SourceCoverage` and `through` to `aggregate` selects current-only mixed calculation. Its explicit interval and independent key schemas are defined in [the API reference](api.md#mixed-source-coverage). Working `bars` stays source-only; returned derived bars are not reinserted. All candidate validation and calculation completes before assigning working state. Neither declaration frame is mutated.

Coverage assertions include no-trade intervals; source-row density cannot prove completeness. Each source row must correspond to a fully covered canonical window. Higher-timeframe rows must be closed at the source watermark. The planner uses the same session geometry as aggregation and lookup, selecting disjoint contained source windows, preferring longer contributions only when they lead to a complete path. Backward reachability over eligible endpoints avoids rejecting a complete cover because an initially longer summary dead-ends. If no compatible path completely covers a requested window, it fails rather than splitting an indivisible summary or inventing prices. A source must occupy a contiguous portion of the requested window’s eligible time. Interleaved repeated-DST subhour summaries require finer source inputs; they are rejected when they cannot form such a cover. Callers supply compatible grains. Source intervals at the same grain may be adjacent but cannot overlap.

Stored grains may overlap for different calculations, such as a prior quarter and prior-week days needed on April 1. Only one representation contributes to each portion of any calculated window. Callers assert that those representations are coherent, consistently adjusted point-in-time data. A minute snapshot overlapping a retained higher-timeframe source is rejected before mutation because the summary might become stale. Supply a corrected coherent source frame and declaration instead. The library does not retrieve children, repair every affected historical row, or advance an archive on rollover.

Canonical schedules and window geometry are reused inside each instance with bounded entry counts and no price/source cache. Cache scope includes session timezone, anchor, schedule identity, timeframe and query range. Calendar changes inside the same callable require a new instance. Session eligibility, breaks, holiday labels, overnight sessions, DST and clipped closes retain the same canonical rules; incompatible source partitions fail explicitly.

Classification needs OHLC and valid timestamp/grouping fields only. Missing volume stays missing, and supplied volume is untouched metadata in both DataFrame and whole-bar scalar results. Aggregate operations continue to require valid numeric volume.
