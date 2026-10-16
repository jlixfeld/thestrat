# API reference

Import `TheStrat` and its configuration/domain types from `thestrat`. Version 2 removes the previous API without compatibility aliases, wrappers or legacy modules. [behavior.md](behavior.md) defines the calculations; the [README](../README.md) contains an executable example and displayed output.

## TheStrat

```python
TheStrat(*, session: Session,
         timeframes: Sequence[Timeframe | str] = DEFAULT_TIMEFRAMES,
         bars: pl.DataFrame | None = None)

aggregate(updates: pl.DataFrame | None = None,
          *, mode: Literal["snapshot"] = "snapshot") -> pl.DataFrame
classify(bars: pl.DataFrame, *, fields: Sequence[Field] | None = None,
         history: int = 3) -> pl.DataFrame
window_starts(timestamp: datetime, *,
              timeframes: Sequence[Timeframe | str] | None = None,
              count: int = 1,
              direction: Literal["backward", "forward"] = "backward",
              inclusive: bool = True) -> pl.DataFrame
```

`classify` is a callable internal component with a `scalar` component, not another public top-level class. `Field` selects `scenario`, `color`, `shape` or `in_force`.

`bars` is an ordinary assignable DataFrame. Assigning or filtering it performs no calculation. Operations validate the frame when consuming it. `aggregate()` calculates the working source minutes, stores source plus aggregate rows in `bars`, and returns only configured target rows. `aggregate(updates)` first applies complete source-minute snapshots by key. An identical repeated snapshot contributes volume once. An error leaves the instance's working frame unchanged.

`classify(frame)` returns the same bar rows with requested classification columns, ordered by symbol, timeframe and timestamp. It neither aggregates nor assigns the instance's working frame. Caller metadata is preserved. Aggregation returns standard bar columns and drops metadata; callers manage separate source-freshness and instrument metadata themselves.

Configured targets are nonempty, unique members of `1min, 5min, 15min, 30min, 1h, 4h, 6h, 12h, 1d, 1w, 1m, 1q, 1y`. `DEFAULT_TIMEFRAMES` contains those thirteen targets. `1m` means calendar month. Source updates are one-minute bars. Source minutes remain visible in `bars` even if `1min` is not a selected return target. A new instance supplies an independent batch workflow.

## Bar frames

| Column | Required value |
| --- | --- |
| `symbol` | Nullable String. If absent, one null-symbol series is created. |
| `timestamp` | Non-null Datetime containing source interval start or aggregate window start. Aware values normalize to UTC; naive values are interpreted as UTC. Output uses UTC microseconds. |
| `timeframe` | Supported timeframe string. Omitted on a source-only input means `1min`. Updates must be `1min`. Mixed minute/aggregate state must identify timeframes. |
| `open`, `high`, `low`, `close` | Finite numeric prices, normalized Float64; `low <= open,close <= high`. Negative prices are allowed. |
| `volume` | Finite nonnegative numeric volume, normalized Float64. Fractional volume is preserved. |

The working key is `(symbol, timeframe, timestamp)`; null symbol is a group. Supply at most one snapshot per key in an input frame. Replacement across calls is supported; duplicate cleanup and provider normalization belong to the caller. Source timestamps must be minute starts, with zero seconds/subseconds. Receipt and transmission times are not source interval identity.

Frames must have a homogeneous session, instrument-label convention, price-adjustment basis and volume unit. Mixed symbols are automatic; sessions or price bases are not inferred from symbols. Typed empty input returns the documented empty schema after validation. Missing market minutes do not create synthetic rows.

An aggregate-only seed can be classified. Updating its window requires the underlying source minutes. The library raises `InsufficientSourceDataError` when unavailable sources prevent accurate aggregation; it cannot prove missing coverage by looking for a trade in every minute. Callers must supply the full retained source set rather than a partial subset presented as complete. No source history survives secretly after caller pruning.

## Classification fields and history

`fields=None` selects all four fields. An empty selector validates/sorts but adds no classifications. Requested existing columns are recalculated; unrequested input columns remain caller data. Only necessary calculations and dependencies run.

| Field | Output columns | Calculation inputs |
| --- | --- | --- |
| `scenario` | `scenario0`…`scenarioN`, nullable String | Current and preceding high/low |
| `color` | `color0`…`colorN`, String with null unavailable history | Open and close |
| `shape` | `shape`, nullable String | Current OHLC |
| `in_force` | `in_force`, nullable Boolean | Current OHLC and prior high/low; scenario/color internally |

`history=N` is a nonnegative integer maximum prior offset. Zero is current; three provides current plus three prior classifications. There is no fixed maximum depth. Scenario/color history is isolated by `(symbol, timeframe)` and comes only from the supplied frame. Current-plus-N scenario values can require N+2 OHLC bars. Shape and in-force are current-only fields. Missing context gives null, not fabricated history.

## Scalar classification

Call `strat.classify.scalar(kind=..., ...)`. Each kind has a typed keyword-only signature and calculates only that current-bar result.

| Kind | Inputs | Result |
| --- | --- | --- |
| `color` | `open_price`, `close_price` | `Color` |
| `scenario` | `high`, `low`, `reference_high`, `reference_low` | `Scenario` |
| `shape` | `open_price`, `high`, `low`, `close_price` | `Shape` or None |
| `in_force` | OHLC prices as above, `reference_high`, `reference_low` | bool |
| `bar` | `bar: BarDict`, optional `prior: BarDict` | `CurrentBar` |

Reference high/low may describe a prior bar or explicit setup levels. `CurrentBar` preserves the bar keys and adds `scenario0`, `color0`, `shape` and `in_force`; it contains no history slots. Missing prior means null scenario/in-force, while color/shape still calculate. `BarDict` and `CurrentBar` are TypedDict types, not Pydantic models. The runtime validates supplied values.

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

## Window lookup

`window_starts` returns `timeframe: String, timestamp: Datetime("us", "UTC")`. `count` is positive and applies per requested timeframe; omitted targets use the instance's configured targets. Backward returns starts at/before the supplied timestamp; forward returns at/after it. `inclusive=False` excludes an exact match. Results are ascending within each requested timeframe.

The query timestamp must be aware. Lookup skips closed sessions and breaks, recognizes calendar range limits, and may return an interval label before eligible open for top-aligned equities. One-minute starts preserve physical UTC identity across a repeated local hour, matching source rows; larger wall-clock targets use the earlier-fold label for merged repeated windows. Lookup performs no scheduling, finalization, clock advancement or broker work.

## Errors and extension

Invalid fields, frames, values and configuration raise useful validation errors. `mode="delta"` raises `NotImplementedError` before state changes, including empty inputs; unknown modes are invalid arguments. `InsufficientSourceDataError` and `ScheduleRangeError` derive from ValueError. Native resource failures may still propagate.

Future classification additions belong behind the existing components. Define inputs, formulas, output types, dependencies, nulls, history requirements and scalar applicability; add expected-result, parity, grouping and performance checks. Explicitly document any change to default fields or whole-bar output. No signal engine, dynamic plugin registry, provider connection or scalar aggregation is provided.
