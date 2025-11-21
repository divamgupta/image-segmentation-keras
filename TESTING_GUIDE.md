# Testing Guide: Basic Models Unit Tests

This guide explains how to run and validate the new unit tests for the `vanilla_encoder` function in `keras_segmentation/models/basic_models.py`.

## Test Overview

The basic models test suite (`test/unit/models/test_basic_models.py`) provides comprehensive coverage of the `vanilla_encoder` function with 6 test cases:

1. **Import Test**: Verifies successful import and function availability
2. **Default Parameters**: Tests encoder with standard 224×224×3 inputs
3. **Custom Dimensions**: Validates behavior with various input sizes and channel counts
4. **Output Shapes**: Confirms correct tensor dimensions at each encoder level
5. **Tensor Types**: Ensures proper Keras tensor objects are returned
6. **Robustness**: Verifies no empty/null outputs

## Running the Tests

### Method 1: pytest (Recommended)

```bash
# Navigate to project root
cd /path/to/image-segmentation-keras

# Run all basic model tests
python -m pytest test/unit/models/test_basic_models.py -v

# Run specific test method
python -m pytest test/unit/models/test_basic_models.py::TestBasicModels::test_vanilla_encoder_default_params -v

# Run with coverage report
python -m pytest test/unit/models/test_basic_models.py --cov=keras_segmentation.models.basic_models --cov-report=term-missing
```

### Method 2: unittest (Direct)

```bash
# Run all tests in the file
python test/unit/models/test_basic_models.py

# Run with verbose output
python -m unittest test.unit.models.test_basic_models -v
```

### Method 3: Manual Test Execution

```python
import unittest
import sys
sys.path.insert(0, '../../../')

from test.unit.models.test_basic_models import TestBasicModels

# Create test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestBasicModels)

# Run tests
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# Check results
print(f"Tests run: {result.testsRun}")
print(f"Failures: {len(result.failures)}")
print(f"Errors: {len(result.errors)}")
```

## Expected Test Results

### With Keras/TensorFlow Available

```
test/unit/models/test_basic_models.py::TestBasicModels::test_vanilla_encoder_import PASSED
test/unit/models/test_basic_models.py::TestBasicModels::test_vanilla_encoder_default_params PASSED
test/unit/models/test_basic_models.py::TestBasicModels::test_vanilla_encoder_custom_dimensions PASSED
test/unit/models/test_basic_models.py::TestBasicModels::test_vanilla_encoder_output_shapes PASSED
test/unit/models/test_basic_models.py::TestBasicModels::test_vanilla_encoder_tensor_types PASSED
test/unit/models/test_basic_models.py::TestBasicModels::test_vanilla_encoder_no_empty_levels PASSED

========================= 6 passed in 2.34s =========================
```

### Without Keras/TensorFlow (CI Environment)

```
test/unit/models/test_basic_models.py::TestBasicModels::test_vanilla_encoder_import SKIPPED
test/unit/models/test_basic_models.py::TestBasicModels::test_vanilla_encoder_default_params SKIPPED
... (all 6 tests skipped)

========================= 6 skipped in 0.21s =========================
```

## Test Dependencies

The tests require:
- **Python 3.6+**
- **unittest** (built-in)
- **Keras & TensorFlow** (optional - tests skip gracefully if unavailable)

## Understanding Test Behavior

### Encoder Level Structure

The `vanilla_encoder` produces 5 levels with these characteristics:

| Level | Spatial Size | Channels | Description |
|-------|-------------|----------|-------------|
| 0     | 112×112    | 64       | First convolution + pooling |
| 1     | 56×56      | 128      | Second convolution + pooling |
| 2-4   | 28×28      | 256      | Three 256-channel levels |

### Channel Ordering

Tests automatically adapt to the `IMAGE_ORDERING` configuration:
- **channels_last**: `(height, width, channels)`
- **channels_first**: `(channels, height, width)`

### Test Data

The tests use these input configurations:
- Default: 224×224×3 (RGB)
- Custom sizes: 128×128, 256×128, 320×240
- Custom channels: 1 (grayscale), 4 (RGBA)

## Debugging Failed Tests

### Common Issues

1. **Import Errors**
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

2. **Keras Unavailable**
   ```bash
   # Install dependencies
   pip install tensorflow keras
   ```

3. **Shape Mismatches**
   - Verify `IMAGE_ORDERING` in `config.py`
   - Check tensor dimensions with `tensor.shape`

### Verbose Debugging

```python
# Add debug prints to understand tensor shapes
from keras_segmentation.models.basic_models import vanilla_encoder
img_input, levels = vanilla_encoder()
print(f"Input shape: {img_input.shape}")
for i, level in enumerate(levels):
    print(f"Level {i} shape: {level.shape}")
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run Basic Models Tests
  run: |
    python -m pytest test/unit/models/test_basic_models.py -v
    python test/unit/models/test_basic_models.py
```

### Coverage Reporting

```bash
# Generate coverage report
python -m pytest test/unit/models/test_basic_models.py \
  --cov=keras_segmentation.models.basic_models \
  --cov-report=html \
  --cov-report=term-missing
```

## Validation Checklist

After running tests, verify:

- [ ] All 6 tests pass (or skip appropriately)
- [ ] No import errors
- [ ] No shape validation failures
- [ ] Tensor types are correct
- [ ] No empty/null outputs
- [ ] Custom dimensions work correctly

## Related Tests

- `test/unit/models/test_keypoint_models.py` - Similar pattern for keypoint models
- `test/test_models.py` - Integration tests for complete models
- `test/integration_test_keypoints.py` - End-to-end keypoint testing
