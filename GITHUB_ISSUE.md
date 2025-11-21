# Add Unit Tests for Basic Models (`vanilla_encoder`)

## Issue Description

The `keras_segmentation` library currently lacks comprehensive unit tests for its core basic model utilities, specifically the `vanilla_encoder` function in `keras_segmentation/models/basic_models.py`. This function is a fundamental building block used by multiple segmentation models (UNet, SegNet, PSPNet, FCN) but has no test coverage.

## Problem Statement

The `vanilla_encoder` function is critical to the library's functionality as it provides the foundational encoder architecture used across different model types. Without proper unit tests:

1. **Reliability Risk**: Changes to the encoder could break multiple dependent models without detection
2. **Regression Prevention**: No automated way to ensure the encoder maintains expected behavior
3. **Documentation Gap**: No validation that the encoder produces expected tensor shapes and structures
4. **Maintenance Burden**: Developers cannot confidently refactor or optimize the encoder

## Current State

- ✅ Keypoint models have comprehensive unit tests (`test/unit/models/test_keypoint_models.py`)
- ✅ Integration tests exist for end-to-end functionality
- ❌ **Missing**: Unit tests for `vanilla_encoder` function
- ❌ **Missing**: Validation of encoder output shapes and tensor types
- ❌ **Missing**: Testing with different input parameters (dimensions, channels)

## Expected Behavior

The `vanilla_encoder` function should:
- Accept parameters: `input_height`, `input_width`, `channels`
- Return a tuple: `(img_input, levels)` where `levels` is a list of 5 encoder tensors
- Handle different channel orderings (channels_first vs channels_last)
- Produce consistent output shapes across different input dimensions

## Proposed Solution

Create comprehensive unit tests covering:

### Test Coverage Requirements
- [ ] Function import and basic instantiation
- [ ] Default parameter behavior
- [ ] Custom input dimensions (height, width, channels)
- [ ] Output tensor shape validation
- [ ] Keras tensor type validation
- [ ] Channel ordering compatibility (channels_first/channels_last)
- [ ] Empty/null output validation

### Test File Location
```
test/unit/models/test_basic_models.py
```

### Example Test Structure
```python
class TestBasicModels(unittest.TestCase):
    def test_vanilla_encoder_import(self):
        # Test successful import

    def test_vanilla_encoder_default_params(self):
        # Test with default parameters

    def test_vanilla_encoder_custom_dimensions(self):
        # Test with various input dimensions

    def test_vanilla_encoder_output_shapes(self):
        # Validate output tensor shapes

    def test_vanilla_encoder_tensor_types(self):
        # Ensure proper Keras tensor types

    def test_vanilla_encoder_no_empty_levels(self):
        # Verify no empty/null outputs
```

## Dependencies

The `vanilla_encoder` function depends on:
- Keras/TensorFlow (for tensor operations)
- `keras_segmentation.models.config.IMAGE_ORDERING` (for channel ordering)

Tests should gracefully skip when dependencies are unavailable.

## Impact

This enhancement will:
- Improve code reliability and maintainability
- Enable confident refactoring of encoder logic
- Provide documentation through executable examples
- Align testing coverage with other model components

## Priority

**Medium** - Core functionality but not blocking current usage. However, essential for long-term maintainability.

## Related Files

- `keras_segmentation/models/basic_models.py` (function to test)
- `keras_segmentation/models/config.py` (IMAGE_ORDERING dependency)
- `test/unit/models/test_keypoint_models.py` (reference test structure)

## Acceptance Criteria

- [ ] All tests pass in CI environment
- [ ] Test coverage includes all public functionality of `vanilla_encoder`
- [ ] Tests follow existing project patterns and conventions
- [ ] Documentation updated to reflect test coverage
- [ ] No regression in existing functionality
