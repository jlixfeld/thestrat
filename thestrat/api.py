"""Explicit snapshot aggregation over visible caller-owned working state."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal, cast

import polars as pl

from thestrat.classifier import Classifier
from thestrat.coverage import SourceCoverage
from thestrat.extension import extend_window, tag_prior, window_stats
from thestrat.frames import KEY, PRICES, SCHEMA, normalize, targets
from thestrat.mixed import calculate_mixed
from thestrat.sessions import Session
from thestrat.types import DEFAULT_TIMEFRAMES, Timeframe
from thestrat.windows import WindowCache


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
        # Prior-snapshot bookkeeping only matters when a loaded row could be extended.
        tracked = summaries.filter(pl.col("timeframe").is_in(list(self.timeframes))).height > 0
        tagged = (tag_prior(sources, incoming) if tracked else self._replaced(sources, incoming)).sort(
            "symbol", "timestamp", nulls_last=False
        )
        outputs = [tagged.select(list(SCHEMA))]
        if tagged.height:
            start, end = (
                cast(datetime, tagged["timestamp"].min()),
                cast(datetime, tagged["timestamp"].max()) + timedelta(minutes=1),
            )
            boundaries, schedule = self._calendar().intervals(
                start, end, tuple(tf for tf in self.timeframes if tf != "1min")
            )
            eligible_intervals = [
                (row["segment_open"], row["segment_open"], row["segment_close"])
                for row in schedule.iter_rows(named=True)
            ]
            tagged = self._mapped(tagged, eligible_intervals).drop("__window")
            outputs = [tagged.select(list(SCHEMA))]
            for tf in self.timeframes:
                if tf == "1min":
                    continue
                mapped = self._mapped(tagged, boundaries[tf])
                if mapped.is_empty():
                    continue
                outputs.append(self._windows_for(tf, window_stats(mapped, tracked=tracked), summaries))
        calculated = pl.concat(outputs)
        # Loaded summaries that no submitted minute touched stay as they are, and so do
        # aggregate-only seeds outside the selected timeframes.
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

    @staticmethod
    def _replaced(sources: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
        if incoming.is_empty():
            return sources
        return pl.concat([sources.join(incoming.select(KEY), on=KEY, how="anti", nulls_equal=True), incoming])

    @staticmethod
    def _windows_for(tf: str, stats: pl.DataFrame, summaries: pl.DataFrame) -> pl.DataFrame:
        """Bars for one timeframe: recalculated unless a loaded row exists, which is extended or kept."""
        if "touched" not in stats.columns:
            return stats.select(
                "symbol",
                pl.col("__window").alias("timestamp"),
                pl.lit(tf).alias("timeframe"),
                *(pl.col(f"new_{n}").alias(n) for n in PRICES),
            )
        held = summaries.filter(pl.col("timeframe") == tf).select(
            "symbol", pl.col("timestamp").alias("__window"), *(pl.col(n).alias(f"held_{n}") for n in PRICES)
        )
        stats = stats.join(held, on=["symbol", "__window"], how="left", nulls_equal=True)
        fresh = stats.filter(pl.col("held_open").is_null())
        bars = fresh.select(
            "symbol",
            pl.col("__window").alias("timestamp"),
            pl.lit(tf).alias("timeframe"),
            *(pl.col(f"new_{n}").alias(n) for n in PRICES),
        )
        extended = [
            {
                "symbol": window["symbol"],
                "timestamp": window["__window"],
                "timeframe": tf,
                **extend_window(window),
            }
            for window in stats.filter(pl.col("held_open").is_not_null() & pl.col("touched")).iter_rows(named=True)
        ]
        return pl.concat([bars, pl.DataFrame(extended, schema=SCHEMA)]) if extended else bars

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
