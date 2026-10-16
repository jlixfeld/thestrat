# Working on thestrat

Read [docs/api.md](docs/api.md) before changing public names, frame schemas or validation. Read [docs/behavior.md](docs/behavior.md) before changing classification, source coverage, rolling updates, sessions or window lookup. Those files describe implemented behavior; keep them and the executable README examples aligned with code.

The public operation owner is `TheStrat`. Preserve explicit calculation, caller-owned pruning and duplicate sanitation, visible assignable working state, automatic symbol/timeframe isolation and vectorized DataFrame classification. Classification returns one scenario, color, shape and in_force value per bar; callers select recent rows with Polars. Mixed aggregation keeps supplied sources separate from returned derived rows and requires independent declared coverage. Classification does not require volume. Snapshot ingestion is implemented; delta requests raise NotImplementedError before mutation.

This is a hard-cut API. Remove obsolete code and tests for obsolete entry points while retaining their meaningful calculation checks through the new interface. Add no aliases, forwarding wrappers, legacy import-error stubs or provider-specific normalization. New bar classifications require documented inputs, dependencies, output types, predecessor/null behavior, expected-result tests and relevant scalar/vector parity.

Use `uv sync --extra test --extra dev` for setup. Verify with `uv run python -m pytest`, `uv run ruff check .`, `uv run ruff format --check .` and `uv run pyright`. Tests belong in `tests/`. Update README output checks whenever its executable example changes. Use real expected values, not snapshots generated solely from the implementation under test.

Keep market-data I/O, broker operations, persistence, scheduling and source freshness with callers. Calendar adapters describe eligibility and boundaries only. Do not infer source completeness from a dense minute grid: missing trade minutes can be valid.
