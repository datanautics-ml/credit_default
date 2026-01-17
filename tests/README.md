# Test Suite for Credit Default Prediction

This directory contains unit tests for the credit default prediction project.

## Test Structure

### `conftest.py`
Shared pytest fixtures:
- `sample_data`: Creates synthetic credit default data for testing
- `temp_dir`: Provides temporary directories for file operations
- `sample_train_test_data`: Pre-split train/test data

### `test_training.py`
Tests for the `ModelTrainer` class:
- ✅ Model initialization
- ✅ Model setup (`_setup_models`, `_setup_scalers`)
- ✅ Data preparation (`prepare_data`)
- ✅ Model training (`train_model`)
- ✅ Model evaluation (`evaluate_model`)
- ✅ Training all models (`train_all_models`)
- ✅ Best model selection (`get_best_model`)
- ✅ Model saving (`save_models`)
- ✅ Error handling for edge cases

### `test_settings.py`
Tests for configuration management:
- ✅ Settings initialization
- ✅ Path creation and validation
- ✅ Environment variable reading
- ✅ Default value verification

### `test_logging.py`
Tests for logging utilities:
- ✅ Logger setup
- ✅ Custom log levels
- ✅ File logging
- ✅ Handler management

### `test_data_preparation.py`
Tests for data preparation logic:
- ✅ Column removal (ID, target)
- ✅ Feature preservation
- ✅ Train-test split ratios
- ✅ Reproducibility

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov=flows --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_training.py
```

### Run specific test class
```bash
pytest tests/test_training.py::TestModelTrainer
```

### Run specific test
```bash
pytest tests/test_training.py::TestModelTrainer::test_train_model_without_optimization
```

### Run with verbose output
```bash
pytest tests/ -v
```

## Test Coverage Goals

- **ModelTrainer**: 80%+ coverage
- **Settings**: 100% coverage
- **Logging**: 90%+ coverage
- **Data Preparation**: 85%+ coverage

## Writing New Tests

When adding new functionality:

1. **Create test file**: `tests/test_<module_name>.py`
2. **Use fixtures**: Leverage `conftest.py` fixtures
3. **Mock external dependencies**: Use `unittest.mock` for MLflow, file I/O, etc.
4. **Test edge cases**: Empty data, missing columns, invalid inputs
5. **Test error handling**: Verify appropriate exceptions are raised

## Example Test Structure

```python
import pytest
from unittest.mock import patch, MagicMock

class TestNewFeature:
    """Test suite for new feature"""
    
    @pytest.fixture
    def fixture_name(self):
        """Setup for tests"""
        return test_data
    
    def test_basic_functionality(self, fixture_name):
        """Test basic use case"""
        result = function_under_test(fixture_name)
        assert result is not None
    
    def test_error_handling(self):
        """Test error cases"""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```
