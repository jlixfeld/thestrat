# API reference

Import `TheStrat` and its configuration/domain types from `thestrat`. Version 2 removes the previous API without compatibility aliases, wrappers or legacy modules. [behavior.md](behavior.md) defines the calculations; the [README](../README.md) contains an executable example and displayed output.

## TheStrat

```python
TheStrat(*, session: Session,
         timeframes: Sequence[Timeframe | str] = DEFAULT_TIMEFRAMES,
         bars: pl.DataFrame | None = None)

aggregate(updates: pl.DataFrame | None = None,
          *, mode: Literal["snapshot"] = "snapshot",
          coverage: SourceCoverage | None = None, through: datetime | None = None) -> pl.DataFrame
classify(bars: pl.DataFrame, *, fields: Sequence[Field] | None = None) -> pl.DataFrame
window(timestamp: datetime | Sequence[datetime], *,
       timeframes: Sequence[Timeframe | str] | None = None,
       before: int = 0, after: int = 0) -> pl.DataFrame
```

`classify` is a callable internal component with a `scalar` component, not another public top-level class. `Field` selects `scenario`, `color`, `shape` or `in_force`.

`bars` is an ordinary assignable DataFrame. Assigning or filtering it performs no calculation. Operations validate the frame when consuming it. Classification validates OHLC, timestamps and grouping; optional volume is unused metadata, not a calculation input. `aggregate()` without coverage calculates the working source minutes, stores source plus aggregate rows in `bars`, and returns only configured timeframe rows. `aggregate(updates)` first applies complete source-minute snapshots by key. An identical repeated snapshot contributes volume once. An error leaves the instance's working frame unchanged.

`classify(frame)` returns the same bar rows with requested classification columns, ordered by symbol, timeframe and timestamp. It neither aggregates nor assigns the instance's working frame. Caller metadata is preserved. Aggregation returns standard bar columns and drops metadata; callers manage separate source-freshness and instrument metadata themselves.

Configured timeframes are nonempty, unique members of `1min, 5min, 15min, 30min, 1h, 4h, 6h, 12h, 1d, 1w, 1m, 1q, 1y`. `DEFAULT_TIMEFRAMES` contains those thirteen timeframes. `1m` means calendar month. Source updates are one-minute bars. Source minutes remain visible in `bars` even if `1min` is not a selected return timeframe. A new instance supplies an independent batch workflow.

## Bar frames

| Column | Required value |
| --- | --- |
| `symbol` | Nullable String. If absent, one null-symbol series is created. |
| `timestamp` | Non-null Datetime containing source interval start or aggregate window start. Aware values normalize to UTC; naive values are interpreted as UTC. Output uses UTC microseconds. |
| `timeframe` | Supported timeframe string. Omitted on a source-only input means `1min`. Updates must be `1min`. Mixed minute/aggregate state must identify timeframes. |
| `open`, `high`, `low`, `close` | Finite numeric prices, normalized Float64; `low <= open,close <= high`. Negative prices are allowed. |
| `volume` | Aggregation requires finite nonnegative numeric volume, normalized Float64; fractional volume is preserved. Classification leaves this optional, unused metadata unchanged when present. |

The working key is `(symbol, timeframe, timestamp)`; null symbol is a group. Supply at most one snapshot per key in an input frame. Replacement across calls is supported; duplicate cleanup and provider normalization belong to the caller. Source timestamps must be minute starts, with zero seconds/subseconds. Receipt and transmission times are not source interval identity.

Frames must have a homogeneous session, instrument-label convention, price-adjustment basis and volume unit. Mixed symbols are automatic; sessions or price bases are not inferred from symbols. Typed empty input returns the documented empty schema after validation. Missing market minutes do not create synthetic rows.

An aggregate-only seed can be classified. In ordinary aggregation without coverage, updating its window requires the underlying source minutes. Mixed aggregation with explicit coverage can instead use complete compatible closed children; see [Mixed source coverage](#mixed-source-coverage). The library raises `InsufficientSourceDataError` when unavailable sources prevent accurate aggregation; it cannot prove missing coverage by looking for a trade in every minute. Callers must supply the full retained source set rather than a partial subset presented as complete. No source history survives secretly after caller pruning.

## Classification fields

`fields=None` selects all four fields. An empty selector validates/sorts but adds no classifications. Requested existing columns are recalculated; unrequested input columns remain caller data. Only necessary calculations and dependencies run.

| Field | Output columns | Calculation inputs |
| --- | --- | --- |
| `scenario` | `scenario`, nullable String | Current and preceding high/low |
| `color` | `color`, String | Open and close |
| `shape` | `shape`, nullable String | Current OHLC |
| `in_force` | `in_force`, nullable Boolean | Current OHLC and prior high/low; scenario/color internally |

Each input bar receives one value per selected field. Scenario and in-force use the preceding bar within `(symbol, timeframe)`; missing predecessors give null. Color and shape need only the current bar. Classification adds no lag columns and accepts no history parameter.

Classify with the required predecessor context before filtering recent rows. Use ordinary Polars operations such as `filter`, `tail` or `group_by("symbol", "timeframe").tail(N)` to inspect recent classifications. Truncating before classification removes context needed by the first retained bar's scenario and in-force.

## Scalar classification

Call `strat.classify.scalar(kind=..., ...)`. Each kind has a typed keyword-only signature and calculates only that current-bar result.

| Kind | Inputs | Result |
| --- | --- | --- |
| `color` | `open_price`, `close_price` | `Color` |
| `scenario` | `high`, `low`, `reference_high`, `reference_low` | `Scenario` |
| `shape` | `open_price`, `high`, `low`, `close_price` | `Shape` or None |
| `in_force` | OHLC prices as above, `reference_high`, `reference_low` | bool |
| `bar` | `bar: BarDict`, optional `prior: BarDict` | `CurrentBar` |

Reference high/low may describe a prior bar or explicit setup levels. `CurrentBar` preserves the bar keys and adds `scenario`, `color`, `shape` and `in_force`. Missing prior means null scenario/in-force, while color/shape still calculate. `BarDict` and `CurrentBar` are TypedDict types, not Pydantic models. Scalar whole-bar classification requires timestamp and OHLC; its volume key is optional unused metadata and is preserved only when supplied. The runtime validates supplied values.

`Color`: green/red/neutral. `Scenario`: 1/2U/2D/3. `Shape`: hammer/shooter. These domain enums derive from StrEnum; DataFrame columns contain their string values.

## Sessions

`Session(timezone: str, anchor_minutes: int, schedule: Schedule)` is frozen. The IANA timezone defines local boundaries; `anchor_minutes` is minutes since local midnight, from 0 through 1439. Its anchor cannot be later than the actual session open. The schedule is callable:

```python
schedule(*, start: datetime, end: datetime) -> pl.DataFrame
```

Bounds are aware UTC with `start < end`. Return all overlapping eligible segments, with full unclipped boundaries, in this column order and UTC microsecond Datetime dtype:

| Column | Meaning |
| --- | --- |
| `session_open` | Actual open of the full session |
| `session_close` | Actual close, including early closes |
| `segment_open` | Start of an eligible continuous segment |
| `segment_close` | Exclusive end of that segment |

Require `session_open <= segment_open < segment_close <= session_close`, sorted nonoverlapping sessions/segments, identical full-session bounds across segments of a session. Breaks are gaps between segments. A known closed range returns a typed empty frame; unsupported ranges raise `ScheduleRangeError`.

`session_for(InstrumentType | str)` and `SESSIONS` provide `equity_us`, `futures_cme` and `crypto`. Equity uses the NYSE calendar and 09:30 Eastern anchor; the futures preset covers CME equity-index sessions with 18:00 Eastern anchor, not every futures product. Crypto is continuous UTC with anchor zero. FX requires an explicit schedule. Preset calendar queries cover 1900–2100; calendar-rule provenance is the installed pandas-market-calendars version recorded in the lockfile.

For top-of-hour equities, construct `Session("America/New_York", 540, session_for("equity_us").schedule)`. The eligibility open stays 09:30 while hourly and longer intraday labels anchor at 09:00. The default anchor 570 gives 09:30. Both aggregation and lookup use the same settings.

## Window geometry

`window` returns membership and neighboring geometry using the same canonical windows as aggregation. Pass one timezone-aware `datetime` or a sequence of them; inputs normalize to UTC. For a Polars timestamp column, pass `frame["timestamp"].to_list()`. Empty sequences return the typed empty result. Input order and duplicates are preserved, followed by requested timeframe order and chronological canonical labels within each input/timeframe. No public input identifier is added.

| Column | Type and meaning |
| --- | --- |
| `input_timestamp` | Datetime("us", "UTC"): the normalized query timestamp |
| `timeframe` | String: requested timeframe |
| `position` | Int64: zero for the containing window, negative for preceding windows, positive for following windows; closest neighbors are -1 and +1 |
| `window_start` | Datetime("us", "UTC"): canonical aggregation label, which can precede eligible open |
| `window_end` | Datetime("us", "UTC"): last eligible segment end, including early closes; not a nominal next-period boundary |
| `segments` | List(Struct({start: Datetime("us", "UTC"), end: Datetime("us", "UTC")})): full sorted eligible half-open intervals for that canonical window |

`before` and `after` are nonnegative integers, default zero, applied per input/timeframe. Booleans and fractional counts are invalid. Omitted `timeframes` uses configured timeframes; a supplied selection must be nonempty, unique and supported. Requested neighbors must be available or the operation raises `ScheduleRangeError`; it does not silently shorten counts. Custom schedules retain their own supported-range errors. Finite schedules without explicit range errors are searched to representable datetime bounds before an unavailable-neighbor error.

Membership uses the eligible physical segments, including repeated daylight-saving-time intervals, rather than comparing the timestamp with a predecessor label. A contained query returns position zero and requested neighboring distinct windows in canonical label order. Outside eligible sessions—including holidays, weekends and breaks—there is no zero row. Preceding windows must have their last eligible end at/before the query; following windows must have their first eligible start strictly after it. A window spanning a break around the query qualifies for neither direction, and the same canonical label is never returned in both directions.

Segments cover the full canonical period even for a query in the middle of a week, month, quarter or year. Calendar-period geometry uses session-open civil dates and configured local anchors, matching aggregation; for futures this can include a Sunday-opened session whose close falls on the following Monday. One-minute windows preserve physical UTC identity across a repeated local hour. Larger wall-clock timeframes merge repeated intervals under their earlier-fold label; the segment list preserves each eligible physical contribution.

Batch queries share schedule fetches and canonical geometry for overlapping timeframe ranges within the call. Long-period completeness does not require generating a year of minute windows. A bounded instance-owned cache reuses canonical geometry and schedules across calls; it stores no source prices/history. Session timezone, anchor and schedule identity changes invalidate it. Timeframe/range keys keep distinct windows separate. Treat calendar rules as immutable during an instance lifetime; use a new instance after changing a schedule in place. Window queries do not aggregate prices, change `bars`, schedule work, finalize bars, advance clocks or contact brokers.

## Errors and extension

Invalid fields, frames, values and configuration raise useful validation errors. `mode="delta"` raises `NotImplementedError` before state changes, including empty inputs; unknown modes are invalid arguments. `InsufficientSourceDataError` and `ScheduleRangeError` derive from ValueError. Native resource failures may still propagate.

Future classification additions belong behind the existing components. Define inputs, formulas, output types, dependencies, nulls, predecessor context and scalar applicability; add expected-result, parity, grouping and performance checks. Explicitly document any change to default fields or whole-bar output. No signal engine, dynamic plugin registry, provider connection or scalar aggregation is provided.

## Mixed source coverage

```python
SourceCoverage(intervals: pl.DataFrame, expected_keys: pl.DataFrame)
aggregate(updates: pl.DataFrame | None = None, *, mode="snapshot",
          coverage: SourceCoverage | None = None,
          through: datetime | None = None) -> pl.DataFrame
```

With `coverage`, aggregation calculates only the configured windows containing `through`. `through` is an aware, eligible **source-minute start**, inclusive: coverage may extend up to `through + one minute`, exclusively. It is not a receipt timestamp. Without coverage, the existing full retained-minute aggregation behavior is unchanged; `through` requires coverage.

`SourceCoverage` is a frozen declaration holding two independent DataFrames:

- `intervals`: `symbol` (nullable String), `timeframe` (supported source grain), `start` and `end` (aware Datetime, normalized to UTC). Each nonempty half-open interval asserts that all trades at that grain are represented, including any no-trade portions. Overlapping declarations within a symbol/grain are rejected; adjacent declarations may jointly cover a window.
- `expected_keys`: `symbol`, `timeframe`, `timestamp`, with unique full source keys. This manifest must exactly match the final source frame after snapshots. An absent expected row is missing data; an interval portion with no expected trade row is explicitly empty. New snapshot keys must already be listed. Declarations and manifests are not silently extended or reconstructed from filtered working state.

Mixed `bars` remains the authoritative supplied source frame after validated minute snapshot replacement; derived rows are returned separately. Callers replace the complete coherent source frame and declaration for rollover or older corrections. Replacing a declaration is a new completeness assertion: the library cannot discover omitted trades from an apparently sparse frame. Stored grains may overlap, but each calculation selects a complete disjoint cover of compatible canonical source windows. Closed higher-timeframe inputs must fit wholly within their declaration and the requested window; a final higher-timeframe bar extending beyond `through` is rejected. Supplied overlapping representations must describe the same point-in-time prices and volume. An incoming minute correction overlapping a retained summary is rejected until the caller replaces the coherent source set. Failed calculations change neither working bars nor coverage frames.

For example, a closed `5min` bar covering 09:30–09:35 plus source minutes at 09:35 and 09:36 can calculate the current `1h` and `5min` bars at 09:36. Declare `5min` coverage `[09:30,09:35)` and `1min` coverage `[09:35,09:37)` and list the three source keys independently. The hourly volume is the sum of all three source rows; current five-minute volume includes only the two source minutes. The README provides executable UTC inputs and outputs.
