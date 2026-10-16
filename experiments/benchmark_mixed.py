"""Run with uv run --no-sync python -m experiments.benchmark_mixed --output PATH."""

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from experiments.mixed_fixture import archive_for, source_year
from experiments.mixed_timeframes import Geometry, MixedCurrent, prepare_seed, reduce_children, replace
from thestrat import TheStrat, session_for
from thestrat.frames import KEY


def measured(operation):
    wall, cpu = time.perf_counter(), time.process_time()
    result = operation()
    return result, {"wall_ms": (time.perf_counter() - wall) * 1000, "cpu_ms": (time.process_time() - cpu) * 1000}


def stats(samples) -> dict:
    result: dict = {"n": len(samples)}
    for name in ("wall_ms", "cpu_ms"):
        values = sorted(sample[name] for sample in samples)
        result[name] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "p95": values[min(len(values) - 1, int(len(values) * 0.95))] if len(values) >= 20 else None,
        }
    return result


def symbols(frame, count):
    return pl.concat([frame.with_columns(pl.lit(f"S{i}").alias("symbol")) for i in range(count)])


def current_full_minutes(source, through, geometry):
    windows = geometry.at(through)
    return pl.concat([reduce_children(source, tf, window) for tf, window in windows.items()]).sort(KEY)


def library_baseline(source, through):
    strat = TheStrat(session=session_for("equity_us"), bars=source)
    full = strat.aggregate()
    keys = strat.window(through).select("timeframe", pl.col("window_start").alias("timestamp"))
    return full.join(keys, on=["timeframe", "timestamp"], how="semi").sort(KEY), full.height


def deep_bytes(value, seen=None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(deep_bytes(k, seen) + deep_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        total += sum(deep_bytes(v, seen) for v in value)
    return total


def run(counts):
    year, generation = measured(source_year)
    through = datetime(2026, 12, 30, 20, 5, tzinfo=UTC)
    source = year.filter(pl.col("timestamp") <= through)
    archive, archive_cost = measured(lambda: archive_for(source))
    geometry = Geometry()
    _, geometry_cost = measured(lambda: geometry.at(through))
    seed, seed_cost = measured(
        lambda: prepare_seed(archive, through, geometry, complete=True, coverage_start=datetime(2026, 1, 1, tzinfo=UTC))
    )
    expected, reference_cost = measured(lambda: library_baseline(source, through))
    oracle, full_rows = expected
    focused_geometry = Geometry()
    assert_frame_equal(current_full_minutes(source, through, focused_geometry), oracle)
    report = {
        "scope": "Synthetic default NYSE, 2026 point-in-time year through Dec 30 15:05 ET; homogeneous price basis",
        "environment": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "platform": platform.platform(),
            "polars_threads": pl.thread_pool_size(),
        },
        "preparation_one_symbol": {
            "generation": generation,
            "archive_aggregation": archive_cost,
            "initial_geometry": geometry_cost,
            "seed_selection": seed_cost,
            "reference_library": reference_cost,
        },
        "rows_one_symbol": {
            "fixture_year": year.height,
            "source_through_watermark": source.height,
            "archive_including_minutes": archive.height,
            "full_library_return": full_rows,
            "mixed_seed": seed.rows.height,
            "mixed_seed_by_timeframe": dict(seed.rows.group_by("timeframe").len().iter_rows()),
        },
        "memory_one_symbol": {
            "source_dataframe_estimated_bytes": source.estimated_size(),
            "seed_dataframe_estimated_bytes": seed.rows.estimated_size(),
            "coverage_keys_estimated_bytes": seed.required_keys.estimated_size(),
            "geometry_python_deep_bytes": deep_bytes((geometry.active, geometry.short)),
        },
        "workloads": [],
        "definitions": {
            "timings": "Milliseconds per whole symbol batch, wall and process CPU; preparation and correctness assertions excluded from update timers. Cached geometry lookup/invalidation and source validation included in candidate calls.",
            "baseline": "Public library rebuilds all historical configured rows then filters the 13 current rows. Focused comparator scans the same year of source minutes but emits only 13 current rows. Both stream separate symbol calls to avoid allocating 100 year copies; candidate batches symbols together.",
            "memory": "DataFrame estimated_size reports visible data buffers; Python deep size estimates active geometry including segment lists. Process high-water RSS includes fixtures, Polars allocator, oracle and all benchmark phases; it is not candidate-only peak memory.",
            "coverage": "Fixture archive declared complete including empty trade intervals. Missing row checks use supplied key manifest, never dense-grid inference. Through watermark assertion is caller-owned.",
            "rollover": "Caller precomputed closed HTF archive is an input. Its cold creation is timed separately. Rollover input is selected outside timer as a simulated source retrieval; measured hydration includes fresh geometry, seed selection, construction and calculate. No database/network cost.",
            "focused_advancing": "One-symbol focused advancing timing includes canonical cached geometry lookup/refresh and 13 reductions; source concatenation and ingestion validation excluded. Candidate new-minute timing additionally includes snapshot normalization, replacement and manifest check. Warm calculation excludes incoming snapshots in both paths.",
            "classification": "Not timed or implemented; predecessor classification context is outside this OHLCV experiment.",
            "p95": "Nearest-rank-style observed percentile only for n>=20; null otherwise. These are descriptive microbenchmarks, not capacity guarantees.",
        },
    }
    for count in counts:
        batch_seed = prepare_seed(
            symbols(seed.rows, count), through, geometry, complete=True, coverage_start=datetime(2026, 1, 1, tzinfo=UTC)
        )
        engine = MixedCurrent(batch_seed, geometry)
        assert_frame_equal(engine.calculate(), symbols(oracle, count).sort(KEY))
        entry: dict = {
            "symbols": count,
            "candidate_dispatch": "one Polars batch",
            "baseline_dispatch": f"{count} sequential single-symbol calls",
            "state_rows": engine.bars.height,
            "state_dataframe_estimated_bytes": engine.bars.estimated_size(),
            "coverage_keys_estimated_bytes": engine.required_keys.estimated_size(),
            "current_output_rows": 13 * count,
        }
        samples = []
        for _ in range(25):
            actual, timing = measured(engine.calculate)
            samples.append(timing)
            assert_frame_equal(actual, symbols(oracle, count).sort(KEY))
        entry["warm_current_calculation"] = stats(samples)
        samples = []
        for i in range(25):
            update = source.tail(1).with_columns(pl.lit(203.0 if i % 2 else 296.0).alias("volume"))
            batch_update = symbols(update, count)
            actual, timing = measured(lambda engine=engine, batch_update=batch_update: engine.calculate(batch_update))
            samples.append(timing)
            corrected = replace(source, update)
            expected_current = current_full_minutes(corrected, through, focused_geometry)
            assert_frame_equal(actual, symbols(expected_current, count).sort(KEY))
        entry["snapshot_replacement"] = stats(samples)
        for name, operation in [
            ("full_library_rebuild", library_baseline),
            ("focused_current_full_minutes", lambda s, t: (current_full_minutes(s, t, focused_geometry), 13)),
        ]:
            samples = []
            for _ in range({1: 5, 10: 3, 100: 1}.get(count, 1)):

                def stream(count=count, operation=operation):
                    last = None
                    for _symbol in range(count):
                        # Identical deterministic values with symbol-normalized fixture;
                        # each logical symbol really executes the full computation.
                        last = operation(source, through)
                    assert last is not None
                    return last

                result, timing = measured(stream)
                samples.append(timing)
                assert_frame_equal(result[0], oracle)
            entry[name] = stats(samples)
        # Ordinary forward minutes; refresh minute/subhour geometry within active hour.
        engine = MixedCurrent(batch_seed, Geometry())
        engine.calculate()  # cold preparation belongs to setup, not steady update
        working = source
        samples = []
        focused_new_samples = []
        for i in range(1, 21):
            new_time = through + timedelta(minutes=i)
            update = year.filter(pl.col("timestamp") == new_time)
            working = pl.concat([working, update])
            batch_update = symbols(update, count)
            actual, timing = measured(
                lambda engine=engine, batch_update=batch_update, new_time=new_time: engine.calculate(
                    batch_update, through=new_time
                )
            )
            samples.append(timing)
            focused_result, focused_timing = measured(
                lambda working=working, new_time=new_time: current_full_minutes(working, new_time, focused_geometry)
            )
            focused_new_samples.append(focused_timing)
            assert_frame_equal(actual, symbols(focused_result, count).sort(KEY))
        entry["new_minute"] = stats(samples)
        entry["focused_advancing_current_one_symbol"] = stats(focused_new_samples)
        # Restore one older hour. Supplied-source retrieval is outside the calculation timer.
        old = datetime(2026, 12, 30, 14, 30, tzinfo=UTC)
        older_update = working.filter(pl.col("timestamp") == old).with_columns(pl.lit(999.0).alias("high"))
        corrected_working = replace(working, older_update)
        restore_source = symbols(
            corrected_working.filter((pl.col("timestamp") >= old) & (pl.col("timestamp") < old + timedelta(hours=1))),
            count,
        )

        def restore(engine=engine, old=old, restore_source=restore_source):
            engine.restore("1h", old, restore_source, complete=True)
            return engine.calculate()

        actual, restore_timing = measured(restore)
        assert_frame_equal(
            actual,
            symbols(
                current_full_minutes(corrected_working, through + timedelta(minutes=20), focused_geometry), count
            ).sort(KEY),
        )
        entry["older_hour_restoration"] = {
            "timing": restore_timing,
            "source_minute_rows": restore_source.height,
            "source_estimated_bytes": restore_source.estimated_size(),
        }
        for rollover_name, rollover_time in [
            ("hour_rollover", datetime(2026, 12, 30, 20, 30, tzinfo=UTC)),
            ("day_rollover", datetime(2026, 12, 31, 14, 30, tzinfo=UTC)),
        ]:
            rollover_source = year.filter(pl.col("timestamp") < rollover_time)
            rollover_archive, archive_roll_cost = measured(
                lambda rollover_source=rollover_source: archive_for(rollover_source)
            )
            new_minute = year.filter(pl.col("timestamp") == rollover_time)
            input_geometry = Geometry()
            rollover_input = prepare_seed(
                pl.concat([rollover_archive, new_minute]),
                rollover_time,
                input_geometry,
                complete=True,
                coverage_start=datetime(2026, 1, 1, tzinfo=UTC),
            ).rows
            batch_input = symbols(rollover_input, count)

            def hydrate(batch_input=batch_input, rollover_time=rollover_time):
                geom = Geometry()
                new_seed = prepare_seed(
                    batch_input, rollover_time, geom, complete=True, coverage_start=datetime(2026, 1, 1, tzinfo=UTC)
                )
                return MixedCurrent(new_seed, geom).calculate()

            actual, rollover_timing = measured(hydrate)
            expected_roll = current_full_minutes(pl.concat([rollover_source, new_minute]), rollover_time, Geometry())
            assert_frame_equal(actual, symbols(expected_roll, count).sort(KEY))
            entry[rollover_name] = {
                "timing": rollover_timing,
                "input_rows": batch_input.height,
                "input_minute_rows": batch_input.filter(pl.col("timeframe") == "1min").height,
                "input_estimated_bytes": batch_input.estimated_size(),
                "fixture_archive_rebuild_one_symbol": archive_roll_cost,
            }
        report["workloads"].append(entry)
        print(
            f"completed {count} symbols: snapshots {entry['snapshot_replacement']['wall_ms']['median']:.2f} ms, new minutes {entry['new_minute']['wall_ms']['median']:.2f} ms",
            flush=True,
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["process_high_water_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="1,10,100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run([int(n) for n in args.symbols.split(",")])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
