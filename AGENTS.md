# Working on thestrat

This package provides Polars OHLCV aggregation and Strat bar classification. Keep these primitives small and portable; preserve public behavior unless the task explicitly changes it. Legacy modules intentionally raise `ImportError` rather than implementing the old API.

Before changing exports or signatures, read [docs/api.md](docs/api.md). Before changing aggregation, sessions, classification or their tests, read [docs/behavior.md](docs/behavior.md), including timezone rules and caller responsibilities. Update those contracts and the runnable README examples when a change affects them.

Use `uv sync --extra test --extra dev` for setup. Verify changes with `uv run python -m pytest`, `uv run ruff check .`, and `uv run ruff format --check .`. The README examples are executed by `tests/test_documentation.py`; preserve their complete inputs and displayed results.

Functions return new frames, but can raise validation errors and Polars exceptions. Keep explicit input validation and useful failure tests. Aggregation does not separate symbols; multi-symbol classification separates symbols but not timeframes. Callers must partition appropriately.
