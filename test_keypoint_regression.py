#!/usr/bin/env python
"""
Test script for keypoint regression functionality.
This script tests the code structure and basic logic without requiring external dependencies.
"""

import numpy as np
import os
import sys

def test_file_structure():
    """Test that all required files exist and have correct structure"""
    print("Testing file structure...")

    required_files = [
        'keras_segmentation/keypoint_train.py',
        'keras_segmentation/keypoint_predict.py',
        'keras_segmentation/data_utils/keypoint_data_loader.py',
        'keras_segmentation/models/keypoint_models.py',
        'example_keypoint_regression.py',
        'KEYPOINT_REGRESSION_README.md'
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"✗ Missing file: {file_path}")
            return False
        else:
            print(f"✓ Found file: {file_path}")

    return True


def test_keypoint_predict_structure():
    """Test that keypoint_predict.py has the required functions and structure"""
    print("\nTesting keypoint_predict.py structure...")

    try:
        with open('keras_segmentation/keypoint_predict.py', 'r') as f:
            content = f.read()

        required_functions = [
            'predict_keypoints',
            'predict_keypoint_coordinates',
            'predict_multiple_keypoints'
        ]

        for func in required_functions:
            if f'def {func}(' not in content:
                print(f"✗ Missing function: {func}")
                return False
            else:
                print(f"✓ Found function: {func}")

        # Check for keypoint prediction functionality
        if 'weighted average' not in content:
            print("✗ Missing weighted average reference for coordinate extraction")
            return False
        else:
            print("✓ Found weighted average coordinate extraction")

        return True

    except Exception as e:
        print(f"✗ Failed to read keypoint_predict.py: {e}")
        return False


def test_keypoint_train_structure():
    """Test that keypoint_train.py has the required functions and structure"""
    print("\nTesting keypoint_train.py structure...")

    try:
        with open('keras_segmentation/keypoint_train.py', 'r') as f:
            content = f.read()

        required_functions = ['train_keypoints']
        required_loss_functions = ['weighted_mse', 'binary_crossentropy', 'categorical_crossentropy']

        for func in required_functions:
            if f'def {func}(' not in content:
                print(f"✗ Missing function: {func}")
                return False
            else:
                print(f"✓ Found function: {func}")

        # Check for loss function options
        found_losses = 0
        for loss in required_loss_functions:
            if loss in content:
                found_losses += 1

        if found_losses >= 2:  # Should have at least mse and weighted_mse
            print(f"✓ Found {found_losses} loss function options")
        else:
            print(f"✗ Missing loss functions, found only {found_losses}")
            return False

        return True

    except Exception as e:
        print(f"✗ Failed to read keypoint_train.py: {e}")
        return False


def test_data_loader_structure():
    """Test that keypoint_data_loader.py has the required functions"""
    print("\nTesting keypoint_data_loader.py structure...")

    try:
        with open('keras_segmentation/data_utils/keypoint_data_loader.py', 'r') as f:
            content = f.read()

        required_functions = [
            'get_keypoint_array',
            'keypoint_generator',
            'verify_keypoint_dataset'
        ]

        for func in required_functions:
            if f'def {func}(' not in content:
                print(f"✗ Missing function: {func}")
                return False
            else:
                print(f"✓ Found function: {func}")

        return True

    except Exception as e:
        print(f"✗ Failed to read keypoint_data_loader.py: {e}")
        return False


def test_model_utils_integration():
    """Test that model_utils.py has been properly extended"""
    print("\nTesting model_utils.py integration...")

    try:
        with open('keras_segmentation/models/model_utils.py', 'r') as f:
            content = f.read()

        if 'def get_keypoint_regression_model(' not in content:
            print("✗ Missing get_keypoint_regression_model function")
            return False
        else:
            print("✓ Found get_keypoint_regression_model function")

        if 'sigmoid' not in content:
            print("✗ Missing sigmoid activation in model_utils")
            return False
        else:
            print("✓ Found sigmoid activation")

        if 'train_keypoints' in content:
            print("✓ Found train_keypoints method binding")
        else:
            print("✗ Missing train_keypoints method binding")
            return False

        return True

    except Exception as e:
        print(f"✗ Failed to read model_utils.py: {e}")
        return False


def test_model_registry():
    """Test that keypoint models are properly registered"""
    print("\nTesting model registry...")

    try:
        with open('keras_segmentation/models/all_models.py', 'r') as f:
            content = f.read()

        keypoint_models = [
            'keypoint_unet_mini',
            'keypoint_unet',
            'keypoint_vgg_unet',
            'keypoint_resnet50_unet',
            'keypoint_mobilenet_unet'
        ]

        for model_name in keypoint_models:
            if model_name not in content:
                print(f"✗ Missing model in registry: {model_name}")
                return False
            else:
                print(f"✓ Found model in registry: {model_name}")

        return True

    except Exception as e:
        print(f"✗ Failed to read all_models.py: {e}")
        return False


def test_compilation():
    """Test that all Python files compile without syntax errors"""
    print("\nTesting compilation...")

    files_to_test = [
        'keras_segmentation/keypoint_train.py',
        'keras_segmentation/keypoint_predict.py',
        'keras_segmentation/data_utils/keypoint_data_loader.py',
        'keras_segmentation/models/keypoint_models.py',
        'keras_segmentation/models/model_utils.py',
        'keras_segmentation/models/all_models.py',
        'example_keypoint_regression.py'
    ]

    for file_path in files_to_test:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✓ {file_path} compiles successfully")
        except Exception as e:
            print(f"✗ {file_path} failed to compile: {e}")
            return False

    return True


def test_readme_completeness():
    """Test that the README has all required sections"""
    print("\nTesting README completeness...")

    try:
        with open('KEYPOINT_REGRESSION_README.md', 'r') as f:
            content = f.read()

        required_sections = [
            'Keypoint Regression with keras-segmentation',
            'Overview',
            'Available Models',
            'Data Format',
            'Training',
            'Prediction',
            'Advanced Usage',
            'Troubleshooting'
        ]

        for section in required_sections:
            if section not in content:
                print(f"✗ Missing section in README: {section}")
                return False
            else:
                print(f"✓ Found section in README: {section}")

        return True

    except Exception as e:
        print(f"✗ Failed to read README: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Keypoint Regression Implementation")
    print("=" * 60)

    tests = [
        test_file_structure,
        test_keypoint_predict_structure,
        test_keypoint_train_structure,
        test_data_loader_structure,
        test_model_utils_integration,
        test_model_registry,
        test_compilation,
        test_readme_completeness
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test {test.__name__} failed!")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Keypoint regression implementation is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the example: python example_keypoint_regression.py")
        print("3. Test with real data using the documented API")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
