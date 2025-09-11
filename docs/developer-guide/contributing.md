# Contributing

Thank you for your interest in contributing to TheStrat! This guide outlines the process for making contributions to this private project.

## Getting Access

This is a private module. Contact the author at `nominal_choroid0y@icloud.com` for:

- Access to the repository
- Contribution guidelines  
- Development discussions
- Feature requests

## Development Setup

Once you have access, set up your development environment:

### 1. Clone and Install

```bash
git clone https://github.com/jlixfeld/thestrat.git
cd thestrat

# Install with all extras
uv sync --extra test --extra dev --extra docs
```

### 2. Verify Setup

```bash
# Run tests
uv run pytest

# Check formatting
uv run ruff format --check .

# Check linting  
uv run ruff check .

# Test documentation build
uv run mkdocs serve
```

### 3. Run Quality Checks

```bash
# Check code quality
uv run ruff check .
uv run ruff format --check .
```

## Code Standards

### Code Style

We use **Ruff** for both linting and formatting:

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .
```

**Configuration** (already in `pyproject.toml`):
- Line length: 120 characters
- Quote style: Double quotes
- Python target: 3.10+

### Type Hints

All public APIs must include type hints:

```python
# Good
def process_data(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Process market data with configuration."""
    return data

# Bad  
def process_data(data, config):
    return data
```

### Documentation

**Docstring Style**: Use Google-style docstrings:

```python
def aggregate_timeframe(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate OHLCV data to specified timeframe.
    
    Args:
        data: Input OHLCV DataFrame with required columns
        timeframe: Target timeframe (e.g., '5m', '1h', '1d')
        
    Returns:
        Aggregated DataFrame with same schema as input
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> data = pd.DataFrame(...)
        >>> result = aggregate_timeframe(data, '5m')
        >>> len(result) < len(data)  # Fewer bars after aggregation
        True
    """
```

**API Documentation**: All public methods need comprehensive docstrings that will be included in the generated API documentation.

## Testing Requirements

### Test Coverage

Maintain **>95% test coverage**:

```bash
# Run with coverage
uv run pytest --cov=thestrat --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Categories

We use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_aggregation_logic():
    """Unit test for aggregation logic."""
    pass

@pytest.mark.integration  
def test_full_pipeline():
    """Integration test for complete pipeline."""
    pass
```

Run specific categories:

```bash
# Unit tests only
uv run pytest -m unit

# Integration tests only  
uv run pytest -m integration

# All tests
uv run pytest
```

### Writing Tests

**Test Structure**: Follow the Arrange-Act-Assert pattern:

```python
def test_timeframe_aggregation():
    # Arrange
    input_data = create_sample_ohlcv_data()
    expected_bars = 20
    
    # Act
    result = aggregate_timeframe(input_data, '5m')
    
    # Assert
    assert len(result) == expected_bars
    assert all(col in result.columns for col in REQUIRED_COLUMNS)
```

**Test Data**: Use fixtures for reusable test data:

```python
@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'open': [100.0] * 100,
        'high': [101.0] * 100,
        'low': [99.0] * 100, 
        'close': [100.5] * 100,
        'volume': [1000] * 100
    })
```

## Contribution Workflow

### 1. Create Feature Branch

```bash
# From main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following our standards
- Add comprehensive tests
- Update documentation if needed
- Ensure all tests pass

### 3. Quality Checks

Before committing, run full quality checks:

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Run full test suite  
uv run pytest --cov=thestrat

# Test documentation
uv run mkdocs build
```

### 4. Commit Changes

Use conventional commit messages:

```bash
# Feature
git commit -m "feat: add multi-timeframe aggregation support"

# Bug fix
git commit -m "fix: handle missing volume data in aggregation"

# Documentation
git commit -m "docs: add examples for forex analysis"

# Tests
git commit -m "test: add integration tests for factory pattern"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request with:
- Clear description of changes
- Link to any related issues
- Test coverage report
- Documentation updates

## Review Process

All contributions go through code review:

1. **Automated Checks**: CI runs tests, linting, coverage
2. **Manual Review**: Code quality, design, documentation  
3. **Testing**: Functionality and edge cases
4. **Documentation**: API docs and user guides updated

## Areas for Contribution

Current focus areas (contact maintainer for details):

- **Performance Optimization**: Polars-first implementations
- **Additional Indicators**: Extended TheStrat patterns
- **Asset Class Support**: New market types  
- **Testing**: Edge cases and integration scenarios
- **Documentation**: More examples and tutorials

## Release Process

Releases follow semantic versioning:

- **Patch** (1.0.1): Bug fixes, documentation
- **Minor** (1.1.0): New features, backwards compatible
- **Major** (2.0.0): Breaking changes

## Getting Help

For contribution questions:

1. **Documentation**: Check existing docs and examples
2. **Issues**: Search existing issues and discussions  
3. **Contact**: Email the maintainer directly
4. **Code Review**: Learn from existing PR reviews

## License

All contributions are subject to the project's private license. By contributing, you agree that your contributions will be licensed under the same terms.

Thank you for helping make TheStrat better!