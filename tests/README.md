# Tests for spatial-grouping-attention

This directory contains comprehensive tests for the spatial-grouping-attention package.

## Test Structure

The test suite is organized into several files:

- `test_spatial_grouping_attention.py` - Main tests for spatial attention classes
- `test_mlp.py` - Focused tests for the MLP module
- `test_utils.py` - Tests for utility functions
- `test_integration.py` - Integration tests and end-to-end workflows
- `conftest.py` - Pytest configuration and shared fixtures

## Test Categories

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests (default for most tests)
- `@pytest.mark.integration` - Integration tests that test multiple components together
- `@pytest.mark.slow` - Slower tests (e.g., those involving forward passes)

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=spatial_grouping_attention --cov-report=html

# Run specific test file
pytest tests/test_mlp.py

# Run specific test function
pytest tests/test_mlp.py::TestMLP::test_mlp_initialization
```

### Using the Test Runner

A convenient test runner script is provided:

```bash
# Run unit tests only
python run_tests.py --unit

# Run integration tests only
python run_tests.py --integration

# Run fast tests only (excludes slow tests)
python run_tests.py --fast

# Run with coverage report
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file test_mlp

# Run with verbose output
python run_tests.py --verbose

# Run tests in parallel
python run_tests.py --parallel auto
```

### Marker-based Selection

```bash
# Run only unit tests
pytest -m "unit"

# Run only integration tests
pytest -m "integration"

# Run only fast tests (exclude slow ones)
pytest -m "not slow"

# Run only slow tests
pytest -m "slow"

# Combine markers
pytest -m "unit and not slow"
```

## Test Coverage

The tests aim to provide comprehensive coverage of:

### MLP Module (`test_mlp.py`)
- Basic initialization with default and custom parameters
- Forward pass with different input shapes and dimensions
- Residual connections and shortcut behavior
- Different activation functions
- Dropout behavior in training vs evaluation modes
- Gradient flow and training loops
- Edge cases and error conditions

### Utility Functions (`test_utils.py`)
- `to_list()` function with scalars, sequences, and nested structures
- `to_tuple()` function with various input types
- Error handling for invalid inputs
- Edge cases like empty sequences and zero dimensions

### Spatial Attention Classes (`test_spatial_grouping_attention.py`)
- Initialization of base `SpatialGroupingAttention` class
- `SparseSpatialGroupingAttention` and `DenseSpatialGroupingAttention` subclasses
- Parameter validation and shape checking
- Component creation (linear layers, norms, etc.)
- String representations
- Abstract method enforcement

### Integration Tests (`test_integration.py`)
- End-to-end workflows (may require full dependencies)
- Memory efficiency and performance characteristics
- Device compatibility (CPU/GPU)
- Training loops and gradient accumulation
- Multi-component interactions

## Dependencies and Mocking

The tests are designed to work even when external dependencies (RoSE, natten) are not installed:

- Core functionality tests run without external dependencies
- Integration tests that require external dependencies are automatically skipped if imports fail
- Mock objects are provided in `conftest.py` for testing when dependencies are missing

## Fixtures

Common test fixtures are provided in `conftest.py`:

- `torch_device` - Provides appropriate torch device for tests
- `random_seed` - Sets reproducible random seeds
- `mock_dependencies` - Mocks external dependencies when not available
- Various tensor fixtures (`small_tensor_2d`, `medium_tensor_2d`, etc.)
- Configuration fixtures (`spatial_configs`)

## Continuous Integration

The tests are configured to run in GitHub Actions with:

- Multiple Python versions (3.10, 3.11, 3.12)
- Multiple platforms (Ubuntu, Windows, macOS)
- Coverage reporting via codecov

## Writing New Tests

When adding new tests:

1. Choose the appropriate test file based on what you're testing
2. Use descriptive test names that explain what is being tested
3. Add appropriate markers (`@pytest.mark.slow` for slow tests)
4. Use fixtures for common setup/teardown
5. Test both happy path and error conditions
6. Include docstrings explaining the test purpose

Example test structure:

```python
class TestNewFeature:
    """Test cases for new feature."""

    def test_basic_functionality(self):
        """Test basic functionality with default parameters."""
        # Test implementation
        pass

    def test_edge_case_handling(self):
        """Test handling of edge cases."""
        # Test implementation
        pass

    @pytest.mark.slow
    def test_performance_characteristics(self):
        """Test performance with large inputs."""
        # Test implementation
        pass
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the package is installed in development mode (`pip install -e .`)

2. **CUDA tests failing**: GPU tests are automatically skipped if CUDA is not available

3. **Dependency errors**: Integration tests requiring external dependencies are automatically skipped if imports fail

4. **Slow tests**: Use `pytest -m "not slow"` to skip time-consuming tests during development

### Debug Mode

Run tests with debugging enabled:

```bash
# Drop into debugger on first failure
pytest --pdb

# Stop on first failure
pytest -x

# More verbose output
pytest -v -s
```
