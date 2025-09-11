# Developer Guide

Welcome to the TheStrat developer guide. This section provides comprehensive information for contributors and developers working with the TheStrat codebase.

## For Maintainers

This project is currently private. Contact the author for access and contribution guidelines.

## Architecture Overview

TheStrat follows a modular design with clear separation of concerns:

### Core Components

- **[Factory](../reference/thestrat/factory.md)** - Component creation and configuration management
- **[Aggregation](../reference/thestrat/aggregation.md)** - OHLCV timeframe data processing
- **[Indicators](../reference/thestrat/indicators.md)** - TheStrat technical analysis implementation
- **[Schemas](../reference/thestrat/schemas.md)** - Configuration models and validation

### Design Patterns

- **Factory Pattern** - Centralized component creation with validation
- **Abstract Base Classes** - Consistent interfaces across components  
- **Configuration-Driven** - Flexible behavior through configuration objects
- **Functional Programming** - Immutable data transformations where possible

## Quick Development Setup

```bash
# Clone repository
git clone https://github.com/jlixfeld/thestrat.git
cd thestrat

# Install all development dependencies
uv sync --extra test --extra dev --extra docs

# Verify installation
uv run pytest
uv run ruff check .
uv run mkdocs serve
```

## Guide Sections

### [Contributing](contributing.md)
Guidelines for making contributions, code style, and pull request process.


## Development Workflow

1. **Setup** - Install dependencies and verify environment
2. **Development** - Write code following project conventions
3. **Testing** - Ensure comprehensive test coverage
4. **Documentation** - Update docs for any API changes
5. **Quality** - Run linting and formatting tools
6. **Review** - Submit changes for review

## Code Quality Standards

- **Test Coverage**: Maintain >95% coverage
- **Code Formatting**: Automated with Ruff
- **Type Hints**: Required for all public APIs
- **Documentation**: Comprehensive docstrings
- **Performance**: Benchmarked critical paths

## Technology Stack

- **Python 3.10+** - Modern Python features
- **Polars** - High-performance data processing
- **Pandas** - Legacy support and interoperability  
- **Pytest** - Testing framework
- **Ruff** - Linting and formatting
- **MkDocs Material** - Documentation

## Getting Help

For development questions:

1. Check existing documentation
2. Review test cases for examples
3. Contact the maintainer directly

## Project Status

**Version**: 1.0.1 - Production/Stable  
**Maintenance**: Active development  
**License**: Private - All rights reserved