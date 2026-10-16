"""Current bars from explicitly complete, compatible source-window covers."""

from datetime import UTC, datetime, timedelta

import polars as pl

from thestrat.coverage import SourceCoverage
from thestrat.frames import KEY, SCHEMA
from thestrat.types import DEFAULT_TIMEFRAMES
from thestrat.windows import WindowCache

Interval = tuple[datetime, datetime]
Tile = tuple[str, datetime, list[Interval]]


def _overlaps(left: list[Interval], right: list[Interval]) -> bool:
    return any(a < d and c < b for a, b in left for c, d in right)


def _covered(pieces: list[Interval], cover: list[Interval]) -> bool:
    ordered = sorted(cover)
    for left, right in pieces:
        cursor = left
        for a, b in ordered:
            if a > cursor:
                break
            if b > cursor:
                cursor = b
            if cursor >= right:
                break
        if cursor < right:
            return False
    return True


def _complete_cover(pieces: list[Interval], tiles: list[Tile]) -> list[Tile]:
    # Eligible time is compressed across calendar closures/breaks. Each source
    # must occupy one contiguous part of that eligible axis; interleaved DST
    # subhour summaries need finer inputs because their endpoints cannot be split.
    options: dict[datetime, list[tuple[Tile, datetime | None]]] = {}
    for tile in tiles:
        child = tile[2]
        if not _covered(child, pieces):
            continue
        first, last = child[0][0], child[-1][1]
        span = [(max(a, first), min(b, last)) for a, b in pieces if a < last and b > first]
        if not _covered(span, child):
            continue
        following = next((max(a, last) for a, b in pieces if b > last), None)
        options.setdefault(first, []).append((tile, following))
    reachable: set[datetime | None] = {None}
    chosen: dict[datetime, tuple[Tile, datetime | None]] = {}
    for point in sorted(options, reverse=True):
        for tile, following in sorted(
            options[point], key=lambda option: (option[0][2][-1][1], option[0][0]), reverse=True
        ):
            if following in reachable:
                chosen[point] = tile, following
                reachable.add(point)
                break
    cursor: datetime | None = pieces[0][0]
    if cursor not in reachable:
        raise ValueError("incomplete compatible source cover; supply complete non-interleaved canonical children")
    result = []
    while cursor is not None:
        tile, cursor = chosen[cursor]
        result.append(tile)
    return result


def _declaration(frame: pl.DataFrame, times: tuple[str, ...]) -> pl.DataFrame:
    required = {"symbol", "timeframe", *times}
    if not isinstance(frame, pl.DataFrame) or set(frame.columns) != required:
        raise ValueError(f"coverage frame requires exactly {sorted(required)}")
    if frame.schema["symbol"] not in (pl.String, pl.Null) or frame.schema["timeframe"] != pl.String:
        raise ValueError("coverage symbol/timeframe must be strings (symbol may be null)")
    for name in times:
        dtype = frame.schema[name]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
            raise ValueError("coverage timestamps must be timezone-aware Datetime")
    result = frame.with_columns(
        pl.col("symbol").cast(pl.String),
        *(pl.col(name).dt.convert_time_zone("UTC").cast(SCHEMA["timestamp"]) for name in times),
    )
    if result.select(pl.any_horizontal(*(pl.col(name).is_null() for name in required - {"symbol"})).any()).item():
        raise ValueError("coverage values must not be null except symbol")
    if result.filter(~pl.col("timeframe").is_in(list(DEFAULT_TIMEFRAMES))).height:
        raise ValueError("unsupported coverage timeframe")
    if result.select(pl.struct(list(required)).is_duplicated().any()).item():
        raise ValueError("duplicate coverage declaration/key")
    return result


def calculate_mixed(
    current: pl.DataFrame,
    incoming: pl.DataFrame,
    coverage: SourceCoverage,
    through: datetime,
    timeframes: tuple[str, ...],
    calendar: WindowCache,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Validate a final source set and reduce it without retaining derived rows."""
    if not isinstance(coverage, SourceCoverage):
        raise ValueError("coverage must be SourceCoverage")
    if not isinstance(through, datetime) or through.tzinfo is None or through.utcoffset() is None:
        raise ValueError("through must be an aware source-minute start")
    if through.second or through.microsecond:
        raise ValueError("through must be an exact source-minute start")
    through = through.astimezone(UTC)
    cutoff = through + timedelta(minutes=1)
    intervals = _declaration(coverage.intervals, ("start", "end"))
    expected = _declaration(coverage.expected_keys, ("timestamp",))
    if intervals.filter((pl.col("start") >= pl.col("end")) | (pl.col("end") > cutoff)).height:
        raise ValueError("coverage requires start < end <= through + one minute")
    candidate = pl.concat([current.join(incoming.select(KEY), on=KEY, how="anti", nulls_equal=True), incoming])
    actual = candidate.select(KEY)
    if expected.join(actual, on=KEY, how="anti", nulls_equal=True).height:
        raise ValueError("missing expected source key; restore the complete source set")
    if actual.join(expected, on=KEY, how="anti", nulls_equal=True).height:
        raise ValueError("source key is not explicitly listed in coverage.expected_keys")
    geometry = calendar.geometry(through, timeframes)
    if geometry.height != len(timeframes):
        raise ValueError("through must fall inside an eligible session segment")
    requested = {
        row["timeframe"]: (
            row["window_start"],
            [(p["start"], min(p["end"], cutoff)) for p in row["segments"] if p["start"] < cutoff],
        )
        for row in geometry.iter_rows(named=True)
    }
    # Identical coverage shapes share planning across symbols within this call.
    plans: dict[tuple, tuple[list[Tile], list[tuple]]] = {}
    mappings = []
    declarations = intervals.partition_by("symbol", as_dict=True, maintain_order=True)
    symbols = set(intervals["symbol"])
    if set(candidate["symbol"]) - symbols:
        raise ValueError("source symbol has no coverage declaration")
    for (symbol,), declarations_for_symbol in declarations.items():
        shape = tuple(
            declarations_for_symbol.select("timeframe", "start", "end").sort("timeframe", "start").iter_rows()
        )
        if shape not in plans:
            by_grain: dict[str, list[Interval]] = {}
            for tf, start, end in shape:
                by_grain.setdefault(tf, []).append((start, end))
            tiles: list[Tile] = []
            for tf, ranges in by_grain.items():
                if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:], strict=False)):
                    raise ValueError("overlapping coverage intervals within a symbol/timeframe")
                windows = {}
                for start, end in ranges:
                    windows.update(calendar.source_windows(tf, start, end))
                for label, pieces in sorted(windows.items()):
                    if not _overlaps(pieces, ranges):
                        continue
                    if not _covered(pieces, ranges):
                        raise ValueError("coverage cuts a source window; supply complete canonical source intervals")
                    if pieces[-1][1] > cutoff:
                        raise ValueError("forming/future higher-timeframe source is not a closed summary")
                    tiles.append((tf, label, pieces))
            selections = []
            for target_tf, (label, pieces) in requested.items():
                for child_tf, child_label, child_pieces in _complete_cover(pieces, tiles):
                    selections.append((child_tf, child_label, target_tf, label, child_pieces[0][0]))
            plans[shape] = tiles, selections
        tiles, selections = plans[shape]
        keys = {(tf, label) for tf, label, _ in tiles}
        symbol_rows = candidate.filter(pl.col("symbol").eq_missing(symbol))
        if any((tf, label) not in keys for tf, label in symbol_rows.select("timeframe", "timestamp").iter_rows()):
            raise ValueError("expected source bar is outside declared canonical coverage")
        updated_minutes = incoming.filter(pl.col("symbol").eq_missing(symbol))["timestamp"].to_list()
        higher = {
            (tf, label)
            for tf, label in current.filter(pl.col("symbol").eq_missing(symbol) & (pl.col("timeframe") != "1min"))
            .select("timeframe", "timestamp")
            .iter_rows()
        }
        if any(
            (tf, label) in higher and any(a <= timestamp < b for a, b in pieces for timestamp in updated_minutes)
            for tf, label, pieces in tiles
        ):
            raise ValueError("snapshot overlaps a retained summary; replace the coherent mixed source set and coverage")
        mappings.extend((symbol, *selection) for selection in selections)
    if not mappings:
        return candidate.sort(KEY, nulls_last=False), pl.DataFrame(schema=SCHEMA)
    mapping = pl.DataFrame(
        mappings,
        schema={
            "symbol": pl.String,
            "timeframe": pl.String,
            "timestamp": SCHEMA["timestamp"],
            "__target": pl.String,
            "__label": SCHEMA["timestamp"],
            "__order": SCHEMA["timestamp"],
        },
        orient="row",
    )
    output = (
        candidate.join(mapping, on=KEY, how="inner", nulls_equal=True)
        .sort("__order")
        .group_by("symbol", "__target", "__label")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        )
        .rename({"__target": "timeframe", "__label": "timestamp"})
        .select(list(SCHEMA))
    )
    order = {tf: i for i, tf in enumerate(timeframes)}
    output = (
        output.with_columns(pl.col("timeframe").replace_strict(order).alias("__order"))
        .sort("symbol", "__order", "timestamp", nulls_last=False)
        .drop("__order")
    )
    return candidate.sort(KEY, nulls_last=False), output
