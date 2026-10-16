# API reference

All names below are exported from `thestrat`. Inputs and calculations are specified in [behavior.md](behavior.md); complete examples are in [README.md](../README.md).

## Aggregation

`TimeframeAggregator()` has no constructor arguments or stored configuration.

```python
aggregate(bars: pl.DataFrame, timeframe: str, equity_offset: bool = False,
          *, session: Session | None = None) -> pl.DataFrame
aggregate_for_instrument(bars: pl.DataFrame, timeframe: str,
                         instrument_type: InstrumentType) -> pl.DataFrame
bucket_key_expr(timeframe: str, session: Session | None) -> pl.Expr
```

The first two are instance methods. `bucket_key_expr` is a free function requiring both arguments (use `None` explicitly for no session). It returns an expression over the `timestamp` column; use `frame.with_columns(bucket_key_expr(...).alias("bucket"))` to inspect boundaries. For session-aware hour-and-above bucketing, pass an aware `Datetime` column. The aggregator handles naive timestamps itself.

Supported target strings: `1min`, `5min`, `15min`, `30min`, `1h`, `4h`, `6h`, `12h`, `1d`, `1w`, `1m`, `1q`, `1y`. `1m` is a month, not a minute. `5sec` exists in `Timeframe` but is not an aggregation target. These are target buckets; the code does not check that source bars are one minute or prevent requests for a smaller target than the source.

Nonempty input requires `timestamp`, `open`, `high`, `low`, `close`, `volume`. Supply Polars `Datetime` timestamps and numeric OHLCV columns. Output has exactly those six columns in that order; additional columns are dropped. Output timestamps identify the first observation in each bucket, while `bucket_key_expr` returns the boundary. Empty input is returned unchanged before timeframe/schema validation.

`aggregate_for_instrument` looks up `SESSIONS[instrument_type]` and calls `aggregate` with that session. An explicit `session` overrides `equity_offset`. The legacy flag affects only `1h`, `4h`, `6h`, `12h`, with a fixed 30-minute shift. `EQUITY_OFFSET_MINUTES = 30` exposes that shift. Prefer an explicit session for local session alignment, especially on multi-hour and calendar targets.

## Sessions

`Session(timezone: str, anchor_minutes: int)` is a frozen dataclass. `timezone` is an IANA timezone name; `anchor_minutes` is the local day boundary measured from midnight. Construction performs no validation. Use valid timezone names and sensible day offsets; downstream Polars operations can reject invalid timezones.

`InstrumentType` is a string enum. `SESSIONS` is a mutable dictionary holding these presets; `session_for(instrument_type: InstrumentType) -> Session` returns its stored session directly.

| Enum member | Value | Timezone | Anchor minutes | Local boundary |
|---|---|---|---:|---|
| `EQUITY_US` | `equity_us` | `America/New_York` | 570 | 09:30 |
| `FUTURES_CME` | `futures_cme` | `America/New_York` | 1080 | 18:00 |
| `CRYPTO` | `crypto` | `UTC` | 0 | 00:00 |
| `FX` | `fx` | `America/New_York` | 1020 | 17:00 |

These are bucket alignment conventions, not exchange calendars. They do not filter overnight data, breaks, weekends, holidays or early closes.

## Classification

```python
classify_bars_df(df: pl.DataFrame) -> pl.DataFrame
classify_bars_multi_symbol(df: pl.DataFrame) -> pl.DataFrame
classify_bar(bar: dict, prior: dict | None) -> ClassifiedBar
classify_color(open_price: float, close_price: float) -> Color
classify_scenario(curr_high: float, curr_low: float,
                  prev_high: float, prev_low: float) -> Scenario
```

`classify_bars_df` requires `open`, `high`, `low`, `close`, already ordered for a single symbol and timeframe. It does not require or sort `timestamp`. `classify_bars_multi_symbol` additionally requires `symbol`, uses separate history per symbol, and expects chronological order within each symbol. It does not partition by timeframe. Partition mixed timeframes before either call.

Both return the input columns plus `scenario0`–`scenario3`, `color0`–`color3`, `shape`, `in_force`, replacing any same-named columns. Scenario/color/shape columns contain strings; `in_force` contains booleans. Missing history is null. Other input columns are preserved. Typed empty frames with required columns can be classified.

`classify_bar` reads all `BarDict` keys from `bar`; `prior`, when present, only needs `high` and `low`. It returns a dictionary, not a dataclass. Current scenario and color are string-enum members, shape is a string or `None`, and in-force is a bool or `None`. History fields 1–3 are always `None`, even with a prior bar. It does not verify that the bars have matching symbol/timeframe or chronological timestamps.

`SHAPE_BODY_ZONE = 0.33` defines the body zone used for hammers and shooters. See [classification rules](behavior.md#classification-rules) for exact thresholds and strict comparisons.

## Domain types

`BarDict` is a `TypedDict` with required keys:

| Key | Annotation | Meaning |
|---|---|---|
| `symbol` | `str` | Instrument identifier; opaque to classification |
| `timestamp` | `datetime` | Observation timestamp |
| `timeframe` | `str` | Caller-provided timeframe label |
| `open`, `high`, `low`, `close` | `float` | OHLC prices |
| `volume` | `float` | Volume in the caller's units |

`ClassifiedBar` extends `BarDict` with `scenario0`–`scenario3: Scenario | None`, `color0: Color`, `color1`–`color3: Color | None`, `shape: str | None`, `in_force: bool | None`. These typing annotations do not validate dictionaries at runtime.

| Enum | Members and values |
|---|---|
| `Color` | `GREEN="green"`, `RED="red"`, `NEUTRAL="neutral"` |
| `Scenario` | `ONE="1"`, `TWO_UP="2U"`, `TWO_DOWN="2D"`, `THREE="3"` |
| `Shape` | `HAMMER="hammer"`, `SHOOTER="shooter"` |
| `Timeframe` | `FIVE_SEC="5sec"`, `ONE_MIN="1min"`, `FIVE_MIN="5min"`, `FIFTEEN_MIN="15min"`, `THIRTY_MIN="30min"`, `ONE_HOUR="1h"`, `FOUR_HOUR="4h"`, `SIX_HOUR="6h"`, `TWELVE_HOUR="12h"`, `ONE_DAY="1d"`, `ONE_WEEK="1w"`, `ONE_MONTH="1m"`, `ONE_QUARTER="1q"`, `ONE_YEAR="1y"` |

All four enums derive from `StrEnum`. `__version__` is the installed package metadata version, falling back to `"0.0.0+unknown"` if package metadata is absent.

## Errors

| Operation | Failure |
|---|---|
| Nonempty `aggregate`, or `bucket_key_expr` | Unsupported timeframe raises `ValueError` |
| Frame classifiers | Missing required columns raise `ValueError` |
| Aggregator / expressions | Missing columns, incompatible types or invalid timezone operations may raise native Python/Polars errors; there is no uniform wrapper |
| `session_for`, `aggregate_for_instrument` | Unknown instrument key raises `KeyError` |
| `classify_bar` | Missing dictionary fields raise `KeyError`; incompatible values can raise native errors |

Prices, OHLC consistency, null/NaN values, duplicates, sorting and bar completeness are not comprehensively validated. Supply clean, ordered data; native null behavior is not a market-data cleaning policy.

## Legacy imports

Importing any of `thestrat.aggregation`, `thestrat.base`, `thestrat.factory`, `thestrat.indicators`, `thestrat.precision`, `thestrat.schemas`, or `thestrat.signals` immediately raises `ImportError`. These are intentional pre-1.0 stubs. Use the current root exports; the package offers no compatibility implementation of the old API.
