# Behavior and limitations

This describes the current implementation. [api.md](api.md) lists signatures, types and failures; [README.md](../README.md) has executable examples.

## Aggregation rules

The aggregator sorts input rows by `timestamp`, assigns bucket keys, and reduces each bucket to first open, maximum high, minimum low, last close and summed volume. The output timestamp is the first observed timestamp, not the bucket boundary. Results are sorted by that timestamp. Equal-timestamp ties have no explicit ordering contract.

Only `timestamp`, `open`, `high`, `low`, `close`, `volume` survive a nonempty aggregation. No grouping by symbol or timeframe occurs; split those inputs yourself. Empty input returns unchanged, including its schema and extra columns, without validating the timeframe. Neither aggregation nor classification changes the supplied DataFrame in place.

There is no gap filling, duplicate removal, resampling interpolation or completeness check. A partial bucket is returned. `1min` still performs grouping and OHLCV reduction; it is not an identity shortcut. Volume units and price adjustment policy belong to the caller.

### Timeframes and timezones

Minute targets are `1min`, `5min`, `15min`, `30min`. Internally these map to Polars minute intervals. `1m` is instead a calendar month. Hour targets are `1h`, `4h`, `6h`, `12h`. Calendar targets are `1d`, `1w`, `1m`, `1q`, `1y`; weeks begin Monday, months on day 1, quarters in January/April/July/October, and years in January.

For `aggregate`, naive `Datetime` timestamps are temporarily marked UTC and restored to naive output. Aware input retains its original timezone in aggregate output. With no session, truncation uses the timestamp column's timezone; aware non-UTC inputs are not converted to UTC. `bucket_key_expr` without a session also directly truncates the original column.

A session affects hour-and-above targets. The expression converts timestamps to the session timezone, strips timezone information, subtracts `anchor_minutes`, truncates, adds the anchor back, and attaches the session timezone again. Its boundary keys are returned in UTC. The aggregate output still carries the first observed timestamp in the input timezone. Direct users of session-aware hour-and-above expressions should supply aware timestamps rather than rely on Polars handling of naive input.

This is an offset calendar, not a trading-date label. For example, a CME daily bucket begins at 18:00 on the calendar day before most of its daytime trading. A CME weekly bucket begins Monday at 18:00 under Monday-based truncation, not Sunday at 18:00. Likewise, monthly and quarterly boundaries use the shifted calendar rather than exchange-specific trading schedules.

Sub-hour targets always truncate directly; the session parameter has no effect. No session applies exchange trading-hour filters. Equity premarket observations remain in the output, and futures breaks are not removed.

The legacy `equity_offset=True` path subtracts a physical 30 minutes, truncates, then adds 30 minutes. It affects only hour targets and operates in the timestamp timezone. It is not equivalent to the 570-minute New York session anchor on every timeframe. An explicit session takes precedence over this flag.

### Daylight saving time

Session hour-and-above arithmetic uses local wall-clock time so boundaries remain anchored across DST changes. A repeated fall-back hour merges into one wall-clock bucket, keyed at the earlier occurrence. A nonexistent spring-forward key is retried one local hour later. That one-hour fallback is the implementation's limit; it is not a general solution for every historical timezone transition or arbitrary custom session.

Calendar buckets can therefore have different elapsed durations across DST. No-session and sub-hour paths use Polars' direct truncation behavior instead of the session-specific repeated-hour/gap rules.

## Classification rules

Comparisons use strict inequalities: matching a prior high or low does not take it out.

| Current range relative to prior | `scenario0` |
|---|---|
| Higher high and lower low | `3` |
| Higher high only | `2U` |
| Lower low only | `2D` |
| Neither | `1` |

Color is green when close exceeds open, red when close is below open, neutral when equal. No prior bar means no scenario or in-force value, but current color and shape can still be calculated.

Shape uses `range = high - low`, body bottom `min(open, close)`, and body top `max(open, close)`. A zero range has no shape. With `SHAPE_BODY_ZONE = 0.33`, body bottom at or above `low + range * 0.67` produces `hammer`; otherwise body top at or below `low + range * 0.33` produces `shooter`. Other bars have no shape. Threshold equality qualifies. Shape does not depend on scenario or color.

In-force is false for scenario `1`. For `2U`, or green `3`, it is whether close exceeds the prior high. For `2D`, or red `3`, it is whether close is below the prior low. Neutral `3` is false. Equality at the threshold is false. This describes the row's close relative to the prior range, not a trade signal, live trigger or order state.

For frame classification, suffix 0 means current row and suffixes 1, 2, 3 are shifted classifications from preceding rows. The first row's scenario is null; that null remains visible when shifted into history. A scalar `classify_bar` call fills only the current classification; all history fields remain `None`.

`classify_bars_df` uses row order without sorting. `classify_bars_multi_symbol` applies shifts per symbol and preserves the input order. Neither partitions by timeframe or verifies chronological ordering. Neither checks prices against a tick size, validates OHLC consistency, normalizes nulls/NaNs, or repairs market data. Supply clean bars and separate timeframes yourself.

## Scope

This package does not fetch data, provide a market calendar, perform broker actions, generate a backtest, maintain live state, calculate indicators or prescribe trading decisions. Session objects and enums describe inputs; their presence does not validate them. See [API errors](api.md#errors) and the repository's [financial disclaimer](../DISCLAIMER.md).
