# Working on thestrat

## Keep It Simple

This library aggregates and classifies the data it is given. Nothing else. It is not opinionated.

- The caller owns correctness: ordering, pruning, duplicates, freshness, and whether the input is right. The library does not detect, guard, refuse or repair bad input beyond schema validation.
- Before adding a rule, check, error type, option, column or abstraction, ask whether the caller can do it. If so, do not add it.
- Solve the stated need with the smallest change. Do not add machinery for cases no one asked for.
- Tests are a few concrete examples with real expected values. Do not build matrices for every way a caller might misuse the library.

Why this is here (2026-11-02): extending loaded higher-timeframe rows grew into watermarks, consistency checks, refusals and a 58-test matrix. About ten lines of aggregation met the need.

Read [docs/api.md](docs/api.md) before changing public names, frame schemas or validation. Read [docs/behavior.md](docs/behavior.md) before changing classification, source coverage, rolling updates, sessions or window lookup. Those files describe implemented behavior; keep them and the executable README examples aligned with code.

The public operation owner is `TheStrat`. Preserve explicit calculation, caller-owned pruning and duplicate sanitation, visible assignable working state, automatic symbol/timeframe isolation and vectorized DataFrame classification. Classification returns one scenario, color, shape and in_force value per bar; callers select recent rows with Polars. Mixed aggregation keeps supplied sources separate from returned derived rows and requires independent declared coverage. Classification does not require volume. Snapshot ingestion is implemented; delta requests raise NotImplementedError before mutation.

This is a hard-cut API. Remove obsolete code and tests for obsolete entry points while retaining their meaningful calculation checks through the new interface. Add no aliases, forwarding wrappers, legacy import-error stubs or provider-specific normalization. New bar classifications require documented inputs, dependencies, output types, predecessor/null behavior, expected-result tests and relevant scalar/vector parity.

Use `uv sync --extra test --extra dev` for setup. Verify with `uv run python -m pytest`, `uv run ruff check .`, `uv run ruff format --check .` and `uv run pyright`. Tests belong in `tests/`. Update README output checks whenever its executable example changes. Use real expected values, not snapshots generated solely from the implementation under test.

Keep market-data I/O, broker operations, persistence, scheduling and source freshness with callers. Calendar adapters describe eligibility and boundaries only. Do not infer source completeness from a dense minute grid: missing trade minutes can be valid.
