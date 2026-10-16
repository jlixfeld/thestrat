"""Explicit snapshot aggregation over visible caller-owned working state."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal, cast

import polars as pl

from thestrat.classifier import Classifier
from thestrat.coverage import SourceCoverage
from thestrat.frames import KEY, SCHEMA, normalize, targets
from thestrat.mixed import calculate_mixed
from thestrat.sessions import Session
from thestrat.types import DEFAULT_TIMEFRAMES, Timeframe
from thestrat.windows import WindowCache


class InsufficientSourceDataError(ValueError):
    """An opaque summary cannot be updated without underlying source minutes."""


class TheStrat:
    def __init__(
        self,
        *,
        session: Session,
        timeframes: Sequence[Timeframe | str] = DEFAULT_TIMEFRAMES,
        bars: pl.DataFrame | None = None,
    ) -> None:
        if not isinstance(session, Session):
            raise ValueError("session must be a Session")
        self.session = session
        self.timeframes = targets(timeframes)
        self.bars = pl.DataFrame(schema=SCHEMA) if bars is None else bars
        self.classify = Classifier()
        self._windows = WindowCache(session)

    def aggregate(
        self,
        updates: pl.DataFrame | None = None,
        *,
        mode: Literal["snapshot"] = "snapshot",
        coverage: SourceCoverage | None = None,
        through: datetime | None = None,
    ) -> pl.DataFrame:
        if mode == "delta":
            raise NotImplementedError("delta ingestion is not implemented")
        if mode != "snapshot":
            raise ValueError("mode must be snapshot")
        if coverage is None and through is not None:
            raise ValueError("through requires explicit SourceCoverage")
        current = normalize(self.bars)
        incoming = normalize(updates, updates=True) if updates is not None else pl.DataFrame(schema=SCHEMA)
        if coverage is not None:
            if through is None:
                raise ValueError("mixed aggregation requires through")
            candidate, output = calculate_mixed(current, incoming, coverage, through, self.timeframes, self._calendar())
            self.bars = candidate
            return output
        sources = current.filter(pl.col("timeframe") == "1min")
        summaries = current.filter(pl.col("timeframe") != "1min")
        # Existing sources, rather than newly submitted updates, establish whether a
        # summary is opaque. A supplied subset cannot establish historical coverage.
        if summaries.height:
            self._check_sources(summaries, sources, incoming)
        if incoming.height:
            sources = pl.concat([sources.join(incoming.select(KEY), on=KEY, how="anti", nulls_equal=True), incoming])
        sources = sources.sort("symbol", "timestamp", nulls_last=False)
        outputs = [sources]
        if sources.height:
            start, end = (
                cast(datetime, sources["timestamp"].min()),
                cast(datetime, sources["timestamp"].max()) + timedelta(minutes=1),
            )
            boundaries, schedule = self._calendar().intervals(
                start, end, tuple(tf for tf in self.timeframes if tf != "1min")
            )
            eligible_intervals = [
                (row["segment_open"], row["segment_open"], row["segment_close"])
                for row in schedule.iter_rows(named=True)
            ]
            sources = self._mapped(sources, eligible_intervals).select(list(SCHEMA))
            outputs = [sources]
            for tf in self.timeframes:
                if tf == "1min":
                    continue
                mapped = self._mapped(sources, boundaries[tf])
                if mapped.is_empty():
                    continue
                grouped = (
                    mapped.sort("timestamp")
                    .group_by("symbol", "__window")
                    .agg(
                        pl.col("open").first(),
                        pl.col("high").max(),
                        pl.col("low").min(),
                        pl.col("close").last(),
                        pl.col("volume").sum(),
                    )
                )
                outputs.append(
                    grouped.rename({"__window": "timestamp"})
                    .with_columns(pl.lit(tf).alias("timeframe"))
                    .select(list(SCHEMA))
                )
        calculated = pl.concat(outputs)
        # Unaffected aggregate-only seeds remain visible for classification; never reduce them.
        if summaries.height:
            summaries = summaries.join(calculated.select(KEY), on=KEY, how="anti", nulls_equal=True)
            calculated = pl.concat([calculated, summaries])
        order = {tf: i for i, tf in enumerate(dict.fromkeys(("1min", *self.timeframes, *DEFAULT_TIMEFRAMES)))}
        calculated = (
            calculated.with_columns(pl.col("timeframe").replace_strict(order).alias("__order"))
            .sort("symbol", "__order", "timestamp", nulls_last=False)
            .drop("__order")
        )
        self.bars = calculated
        return calculated.filter(pl.col("timeframe").is_in(list(self.timeframes)))

    def _calendar(self) -> WindowCache:
        identity = (self.session.timezone, self.session.anchor_minutes, id(self.session.schedule))
        if self._windows.identity != identity:
            self._windows = WindowCache(self.session)
        return self._windows

    @staticmethod
    def _mapped(sources: pl.DataFrame, intervals: list[tuple[datetime, datetime, datetime]]) -> pl.DataFrame:
        if not intervals or sources.is_empty():
            return sources.with_columns(pl.lit(None, dtype=SCHEMA["timestamp"]).alias("__window")).clear()
        table = (
            pl.DataFrame(
                intervals,
                schema={"__window": SCHEMA["timestamp"], "__left": SCHEMA["timestamp"], "__right": SCHEMA["timestamp"]},
                orient="row",
            )
            .unique()
            .sort("__left")
        )
        return (
            sources.sort("timestamp")
            .join_asof(table, left_on="timestamp", right_on="__left", strategy="backward")
            .filter(pl.col("timestamp") < pl.col("__right"))
            .drop("__left", "__right")
        )

    def _check_sources(self, summaries: pl.DataFrame, sources: pl.DataFrame, incoming: pl.DataFrame) -> None:
        def coverage(frame: pl.DataFrame) -> set[tuple]:
            if frame.is_empty():
                return set()
            start, end = (
                cast(datetime, frame["timestamp"].min()),
                cast(datetime, frame["timestamp"].max()) + timedelta(minutes=1),
            )
            boundaries, _ = self._calendar().intervals(start, end, tuple(tf for tf in self.timeframes if tf != "1min"))
            keys = set()
            for tf in self.timeframes:
                if tf != "1min":
                    keys.update(
                        (symbol, tf, label)
                        for symbol, label in self._mapped(frame, boundaries[tf])
                        .select("symbol", "__window")
                        .unique()
                        .iter_rows()
                    )
            return keys

        existing, affected = coverage(sources), coverage(incoming)
        for row in summaries.iter_rows(named=True):
            key = tuple(row[name] for name in KEY)
            if row["timeframe"] in self.timeframes and key not in existing and (incoming.is_empty() or key in affected):
                raise InsufficientSourceDataError(
                    f"symbol={row['symbol']!r} window={row['timeframe']}@{row['timestamp']}: underlying retained source minutes unavailable"
                )

    def window(
        self,
        timestamp: datetime | Sequence[datetime],
        *,
        timeframes: Sequence[Timeframe | str] | None = None,
        before: int = 0,
        after: int = 0,
    ) -> pl.DataFrame:
        selected = self.timeframes if timeframes is None else targets(timeframes)
        for name, value in (("before", before), ("after", after)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        return self._calendar().geometry(timestamp, selected, before, after)
