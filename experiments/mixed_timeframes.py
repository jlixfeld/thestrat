"""NYSE-only experiment. No public API; callers own archives and completeness.

State contains current-hour minutes and closed children needed for current bars.
Coverage is an explicit assertion about the supplied source set, not a dense grid.
"""

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from thestrat import TheStrat, session_for
from thestrat.frames import KEY, SCHEMA, normalize
from thestrat.windows import windows as canonical_windows

TIERS = ("1min", "1h", "1d", "1m", "1q")
CHILDREN = {
    "1min": "1min",
    "5min": "1min",
    "15min": "1min",
    "30min": "1min",
    "1h": "1min",
    "4h": "1h",
    "6h": "1h",
    "12h": "1h",
    "1d": "1h",
    "1w": "1d",
    "1m": "1d",
    "1q": "1m",
    "1y": "1q",
}


class CoverageError(ValueError):
    """Restore complete children before recomputing a collapsed partition."""


class Geometry:
    """Bounded cache: one active canonical window per timeframe, shared by symbols."""

    def __init__(self):
        self.strat = TheStrat(session=session_for("equity_us"))
        self.active: dict[str, dict] = {}
        self.short: dict[str, list] = {}
        self.short_hour: datetime | None = None

    def at(self, timestamp: datetime) -> dict[str, dict]:
        missing = [
            tf
            for tf in CHILDREN
            if tf not in ("1min", "5min", "15min", "30min")
            if tf not in self.active
            or not any(segment["start"] <= timestamp < segment["end"] for segment in self.active[tf]["segments"])
        ]
        if missing:
            found = self.strat.window(timestamp, timeframes=missing)
            if found.height != len(missing):
                raise ValueError("experiment requires an eligible NYSE minute")
            self.active.update({row["timeframe"]: row for row in found.iter_rows(named=True)})
        hour = self.active["1h"]
        if self.short_hour != hour["window_start"]:
            day = self.active["1d"]
            schedule = pl.DataFrame(
                [
                    {
                        "session_open": day["segments"][0]["start"],
                        "session_close": day["window_end"],
                        "segment_open": segment["start"],
                        "segment_close": segment["end"],
                    }
                    for segment in day["segments"]
                ]
            )
            short = canonical_windows(
                self.strat.session,
                hour["window_start"],
                hour["window_end"],
                ("1min", "5min", "15min", "30min"),
                schedule=schedule,
            )
            self.short = {
                tf: [entry for entry in entries if hour["window_start"] <= entry[0] < hour["window_end"]]
                for tf, entries in short.items()
            }
            self.short_hour = hour["window_start"]
        for tf, entries in self.short.items():
            label, left, right = next(entry for entry in entries if entry[1] <= timestamp < entry[2])
            self.active[tf] = {"window_start": label, "window_end": right, "segments": [{"start": left, "end": right}]}
        return dict(self.active)


def replace(frame: pl.DataFrame, updates: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([frame.join(updates.select(KEY), on=KEY, how="anti", nulls_equal=True), updates])


@dataclass
class Seed:
    rows: pl.DataFrame
    required_keys: pl.DataFrame
    through: datetime
    hour_start: datetime
    # Caller asserts ALL trades through this watermark were used, including
    # explicitly empty windows. Required keys detect accidental dropped rows.
    complete: bool
    coverage_start: datetime


def prepare_seed(
    archive: pl.DataFrame, through: datetime, geometry: Geometry, *, complete: bool, coverage_start: datetime
) -> Seed:
    """Select a bounded seed from caller-owned, point-in-time aggregate history.

    Archive must be computed from the complete minute source through `through`,
    never from final future bars. No archive reference is retained by the engine.
    """
    if not complete:
        raise CoverageError("caller must declare complete point-in-time archive coverage")
    windows = geometry.at(through)
    start = {tf: row["window_start"] for tf, row in windows.items()}
    if coverage_start > min(start["1y"], start["1w"]):
        raise CoverageError("coverage start must include the year and any previous-year days in the current week")
    selections = [
        ("1min", start["1h"], through, True),
        ("1h", start["1d"], start["1h"], False),
        ("1d", min(start["1w"], start["1m"]), start["1d"], False),
        ("1m", start["1q"], start["1m"], False),
        ("1q", start["1y"], start["1q"], False),
    ]
    parts = [
        archive.filter(
            (pl.col("timeframe") == tf)
            & (pl.col("timestamp") >= left)
            & ((pl.col("timestamp") <= right) if inclusive else (pl.col("timestamp") < right))
        )
        for tf, left, right, inclusive in selections
    ]
    rows = pl.concat(parts).select(list(SCHEMA))
    return Seed(rows, rows.select(KEY), through, start["1h"], complete, coverage_start)


def reduce_children(children: pl.DataFrame, tf: str, window: dict) -> pl.DataFrame:
    # Default NYSE child labels align at the open; no child straddles a target.
    selected = children.filter(
        (pl.col("timestamp") >= window["window_start"]) & (pl.col("timestamp") < window["window_end"])
    )
    return (
        selected.sort("timestamp")
        .group_by("symbol")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        )
        .with_columns(
            pl.lit(tf).alias("timeframe"),
            pl.lit(window["window_start"], dtype=SCHEMA["timestamp"]).alias("timestamp"),
        )
        .select(list(SCHEMA))
    )


class MixedCurrent:
    """Bounded, visible Polars state; one NYSE session and watermark per batch."""

    def __init__(self, seed: Seed, geometry: Geometry):
        if not seed.complete:
            raise CoverageError("incomplete seed")
        self.bars = seed.rows
        self.required_keys = seed.required_keys
        self.through = seed.through
        self.hour_start = seed.hour_start
        self.geometry = geometry

    def calculate(self, updates: pl.DataFrame | None = None, *, through: datetime | None = None) -> pl.DataFrame:
        through = self.through if through is None else through
        windows = self.geometry.at(through)
        if windows["1h"]["window_start"] != self.hour_start:
            raise CoverageError("hour rollover requires a new bounded seed from closed summaries and new-hour minutes")
        if through < self.through:
            raise ValueError("watermark cannot move backward")
        if self.required_keys.join(self.bars.select(KEY), on=KEY, how="anti", nulls_equal=True).height:
            raise CoverageError("declared child row missing; restore source partition")
        candidate = self.bars
        if updates is not None:
            updates = normalize(updates, updates=True)
            if updates.filter((pl.col("timestamp") < self.hour_start) | (pl.col("timestamp") > through)).height:
                raise CoverageError("minute outside retained hour; restore its collapsed partition")
            # Eligibility remains canonical, including sparse valid minute sets.
            eligible = pl.any_horizontal(
                *[
                    (pl.col("timestamp") >= segment["start"]) & (pl.col("timestamp") < segment["end"])
                    for segment in windows["1h"]["segments"]
                ]
            )
            if updates.filter(~eligible).height:
                raise ValueError("ineligible source minute")
            candidate = replace(candidate, updates)
        levels = {tf: candidate.filter(pl.col("timeframe") == tf) for tf in TIERS}
        outputs = []
        for tf, child in CHILDREN.items():
            bar = reduce_children(levels[child], tf, windows[tf])
            outputs.append(bar)
            if tf in TIERS and tf != "1min":
                levels[tf] = pl.concat([levels[tf], bar])
        self.bars = candidate
        self.required_keys = candidate.select(KEY)
        self.through = through
        return pl.concat(outputs).sort(KEY)

    def restore(self, tf: str, label: datetime, minutes: pl.DataFrame, *, complete: bool) -> pl.DataFrame:
        """Replace one collapsed partition using its complete minute source set.

        Restores an hour/day/month/quarter already held by this current plan.
        Returns the restored summary. `calculate` then rebuilds current ancestors.
        No source archive survives in the engine.
        """
        if not complete or tf not in TIERS[1:]:
            raise CoverageError("complete source partition declaration required")
        if self.required_keys.join(self.bars.select(KEY), on=KEY, how="anti", nulls_equal=True).height:
            raise CoverageError("declared child row missing; restore complete seed")
        existing = self.bars.filter((pl.col("timeframe") == tf) & (pl.col("timestamp") == label))
        if existing.is_empty():
            raise CoverageError("partition not held by current plan; hydrate a plan for the affected timestamp")
        normalized = normalize(minutes, updates=True)
        if normalized.is_empty():
            raise CoverageError("experiment restoration requires a nonempty partition")
        higher = TIERS[TIERS.index(tf) + 1 :]
        if higher:
            ancestors = self.geometry.strat.window(normalized["timestamp"][0], timeframes=higher).select(
                "timeframe", pl.col("window_start").alias("timestamp")
            )
            if self.bars.join(ancestors, on=["timeframe", "timestamp"], how="semi").height:
                raise CoverageError(
                    "overlapping retained ancestor requires a fresh complete seed with corrected summaries"
                )
        # Closed hours/days cannot overlap the newer retained minutes/hours.
        # Closed months/quarters can overlap prior-week daily rows.
        if tf in ("1m", "1q"):
            partition = self.geometry.strat.window(normalized["timestamp"][0], timeframes=[tf]).row(0, named=True)
            descendants = self.bars.filter(
                pl.col("timeframe").is_in(TIERS[: TIERS.index(tf)])
                & (pl.col("timestamp") >= partition["window_start"])
                & (pl.col("timestamp") < partition["window_end"])
            )
            if descendants.height:
                raise CoverageError(
                    "overlapping retained descendant requires a fresh complete seed with corrected summaries"
                )
        rebuilt = TheStrat(session=session_for("equity_us"), timeframes=[tf], bars=normalized).aggregate()
        if rebuilt.filter(pl.col("timestamp") != label).height or set(rebuilt["symbol"]) != set(existing["symbol"]):
            raise CoverageError("restored sources must cover exactly the named partition and symbols")
        self.bars = replace(self.bars, rebuilt)
        return rebuilt
