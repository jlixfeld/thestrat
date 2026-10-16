"""Measure the supported public mixed-source API separately from the historical prototype."""

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from experiments.mixed_fixture import archive_for, source_year
from thestrat import SourceCoverage, TheStrat, session_for
from thestrat.frames import KEY


def measure(call):
    wall, cpu = time.perf_counter(), time.process_time()
    output = call()
    return output, {"wall_ms": (time.perf_counter() - wall) * 1000, "cpu_ms": (time.process_time() - cpu) * 1000}


def summary(samples) -> dict:
    result: dict = {"n": len(samples)}
    for metric in ("wall_ms", "cpu_ms"):
        values = sorted(row[metric] for row in samples)
        result[metric] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return result


def batch(frame, count):
    return pl.concat([frame.with_columns(pl.lit(f"S{i}").alias("symbol")) for i in range(count)])


def equal(left, right):
    assert_frame_equal(left.sort(KEY), right.sort(KEY), check_exact=True)


def source_inputs(archive, through):
    # Simulates caller retrieval bounds using the supported canonical window API.
    windows = TheStrat(session=session_for("equity_us")).window(through)
    starts = dict(windows.select("timeframe", "window_start").iter_rows())
    ranges = [
        ("1min", starts["1h"], through + timedelta(minutes=1)),
        ("1h", starts["1d"], starts["1h"]),
        ("1d", min(starts["1w"], starts["1m"]), starts["1d"]),
        ("1m", starts["1q"], starts["1m"]),
        ("1q", starts["1y"], starts["1q"]),
    ]
    ranges = [(tf, start, end) for tf, start, end in ranges if start < end]
    rows = pl.concat(
        [
            archive.filter((pl.col("timeframe") == tf) & (pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
            for tf, start, end in ranges
        ]
    )
    intervals = pl.DataFrame(
        [("S0", tf, start, end) for tf, start, end in ranges],
        schema={
            "symbol": pl.String,
            "timeframe": pl.String,
            "start": pl.Datetime("us", "UTC"),
            "end": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )
    return rows, intervals


def focused(source, geometry):
    # Source fixture is already session-eligible; geometry determines each label.
    outputs = []
    for window in geometry.iter_rows(named=True):
        selected = source.filter(
            (pl.col("timestamp") >= window["window_start"]) & (pl.col("timestamp") < window["window_end"])
        )
        outputs.append(
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
                pl.lit(window["timeframe"]).alias("timeframe"), pl.lit(window["window_start"]).alias("timestamp")
            )
            .select(source.columns)
        )
    return pl.concat(outputs)


def public_baseline(source, through):
    strat = TheStrat(session=session_for("equity_us"), bars=source)
    result = strat.aggregate()
    keys = strat.window(through).select("timeframe", pl.col("window_start").alias("timestamp"))
    return result.join(keys, on=["timeframe", "timestamp"], how="semi")


def run(counts):
    year, generation = measure(source_year)
    through = datetime(2026, 12, 30, 20, 5, tzinfo=UTC)
    source = year.filter(pl.col("timestamp") <= through)
    archive, archive_cost = measure(lambda: archive_for(source))
    (rows, intervals), retrieval_cost = measure(lambda: source_inputs(archive, through))
    geometry_owner = TheStrat(session=session_for("equity_us"))
    geometry, geometry_cost = measure(lambda: geometry_owner.window(through))
    oracle = public_baseline(source, through)
    equal(focused(source, geometry), oracle)
    report = {
        "implementation": "TheStrat.aggregate(coverage=SourceCoverage(...), through=...)",
        "environment": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "threads": pl.thread_pool_size(),
            "platform": platform.platform(),
        },
        "source_minutes_per_symbol": source.height,
        "source_rows_per_symbol": rows.height,
        "preparation_one_symbol": {
            "generation": generation,
            "archive_creation": archive_cost,
            "simulated_retrieval_and_bound_geometry": retrieval_cost,
            "focused_geometry": geometry_cost,
        },
        "definitions": {
            "candidate": "Whole batch timings include normalization, independent manifest/coverage validation, closed canonical source planning, geometry lookup/reuse, Polars reductions and atomic source assignment. No prototype runtime used.",
            "new_minute": "20 strictly advancing minute keys; caller creates extended exclusive intervals and final manifest before timing. No missing-row inference or automatic coverage extension.",
            "baseline": "Public full-minute baseline emits historical rows before current selection. Streamed baselines execute one real call per normalized symbol, avoiding100 source-year copies. Focused comparator has cached supported geometry and calculates only13 current rows; source ingestion is excluded.",
            "memory": "Polars estimated_size for caller source/coverage frames only; excludes internal cached geometry, allocator overhead and peak RSS. No isolated candidate peak-memory claim.",
            "rollover": "Caller-owned archive creation and simulated bounded retrieval reported separately. Timed public hydration includes cold internal geometry and source planning. No database/network timing.",
            "correctness": "Exact synthetic float equality checked outside timers for every candidate sample and comparator result. Source values use integer/quarter prices and integral volumes.",
            "limits": "Synthetic homogeneous NYSE data; no classification or external I/O. Multi-symbol streamed comparisons are not equal-batching speed ratios; one equally batched10-symbol focused comparison is included. Desktop activity uncontrolled; n1 has no percentile meaning.",
        },
        "workloads": [],
    }
    for count in counts:
        bars, ranges = batch(rows, count), batch(intervals, count)
        coverage = SourceCoverage(ranges, bars.select(KEY))
        strat = TheStrat(session=session_for("equity_us"), bars=bars)
        actual, cold = measure(
            lambda strat=strat, coverage=coverage: strat.aggregate(coverage=coverage, through=through)
        )
        equal(actual, batch(oracle, count))
        entry: dict = {
            "symbols": count,
            "cold_public_calculation": cold,
            "source_rows": bars.height,
            "source_estimated_bytes": bars.estimated_size(),
            "intervals_estimated_bytes": ranges.estimated_size(),
            "manifest_estimated_bytes": coverage.expected_keys.estimated_size(),
        }
        samples = []
        for _ in range(10):
            actual, cost = measure(
                lambda strat=strat, coverage=coverage: strat.aggregate(coverage=coverage, through=through)
            )
            samples.append(cost)
            equal(actual, batch(oracle, count))
        entry["warm_calculation"] = summary(samples)
        samples = []
        for i in range(20):
            update = source.tail(1).with_columns(pl.lit(203.0 if i % 2 else 296.0).alias("volume"))
            updates = batch(update, count)
            actual, cost = measure(
                lambda strat=strat, coverage=coverage, updates=updates: strat.aggregate(
                    updates, coverage=coverage, through=through
                )
            )
            samples.append(cost)
            corrected = pl.concat([source.head(source.height - 1), update])
            equal(actual, batch(focused(corrected, geometry), count))
        entry["snapshot_replacement"] = summary(samples)
        for name, operation in [
            ("public_full_minute_streamed", public_baseline),
            ("focused_current_streamed", lambda s, t: focused(s, geometry)),
        ]:
            samples = []
            for _ in range(3 if count == 1 else 1):

                def stream(count=count, operation=operation):
                    result = oracle
                    for _symbol in range(count):
                        result = operation(source, through)
                    return result

                result, cost = measure(stream)
                samples.append(cost)
                equal(result, oracle)
            entry[name] = summary(samples)
        if count <= 10:
            batched_minutes = batch(source, count)
            samples = []
            for _ in range(3):
                actual, cost = measure(lambda batched_minutes=batched_minutes: focused(batched_minutes, geometry))
                samples.append(cost)
                equal(actual, batch(oracle, count))
            entry["focused_current_equal_batch"] = summary(samples)
        strat = TheStrat(session=session_for("equity_us"), bars=bars)
        strat.aggregate(coverage=coverage, through=through)
        working = source
        samples = []
        for i in range(1, 21):
            stamp = through + timedelta(minutes=i)
            update = year.filter(pl.col("timestamp") == stamp)
            updates = batch(update, count)
            coverage = SourceCoverage(
                ranges.with_columns(
                    pl.when(pl.col("timeframe") == "1min")
                    .then(pl.lit(stamp + timedelta(minutes=1)))
                    .otherwise(pl.col("end"))
                    .alias("end")
                ),
                pl.concat([coverage.expected_keys, updates.select(KEY)]),
            )
            actual, cost = measure(
                lambda strat=strat, coverage=coverage, updates=updates, stamp=stamp: strat.aggregate(
                    updates, coverage=coverage, through=stamp
                )
            )
            samples.append(cost)
            working = pl.concat([working, update])
            equal(actual, batch(focused(working, geometry_owner.window(stamp)), count))
        entry["new_minute"] = summary(samples)
        for label, stamp in [
            ("hour_reload", datetime(2026, 12, 30, 20, 30, tzinfo=UTC)),
            ("day_reload", datetime(2026, 12, 31, 14, 30, tzinfo=UTC)),
        ]:
            retained = year.filter(pl.col("timestamp") <= stamp)
            reload_archive, archive_timing = measure(lambda retained=retained: archive_for(retained))
            (reload_rows, reload_ranges), retrieval = measure(
                lambda reload_archive=reload_archive, stamp=stamp: source_inputs(reload_archive, stamp)
            )
            reload_rows, reload_ranges = batch(reload_rows, count), batch(reload_ranges, count)
            owner = TheStrat(session=session_for("equity_us"), bars=reload_rows)
            declaration = SourceCoverage(reload_ranges, reload_rows.select(KEY))
            actual, timing = measure(
                lambda owner=owner, declaration=declaration, stamp=stamp: owner.aggregate(
                    coverage=declaration, through=stamp
                )
            )
            equal(actual, batch(focused(retained, geometry_owner.window(stamp)), count))
            entry[label] = {
                "public_calculation": timing,
                "caller_archive_one_symbol": archive_timing,
                "caller_retrieval_one_symbol": retrieval,
                "input_rows": reload_rows.height,
                "input_minute_rows": reload_rows.filter(pl.col("timeframe") == "1min").height,
            }
        report["workloads"].append(entry)
        print(
            f"supported {count} symbols: snapshot {entry['snapshot_replacement']['wall_ms']['median']:.2f}ms, new {entry['new_minute']['wall_ms']['median']:.2f}ms",
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="1,10,100")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run([int(v) for v in args.symbols.split(",")]), indent=2) + "\n")


if __name__ == "__main__":
    main()
