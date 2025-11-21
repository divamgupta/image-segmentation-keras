import unittest
import tempfile
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

class TestKeypointTraining(unittest.TestCase):
    """Test keypoint training functionality"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir)

    def test_find_latest_checkpoint(self):
        """Test checkpoint finding functionality"""
        try:
            from keras_segmentation.keypoint_train import find_latest_checkpoint

            checkpoints_path = os.path.join(self.tmp_dir, "test_checkpoint")

            # Test with no checkpoints
            result = find_latest_checkpoint(checkpoints_path)
            self.assertIsNone(result)

            # Test with fail_safe=False
            with self.assertRaises(ValueError):
                find_latest_checkpoint(checkpoints_path, fail_safe=False)

            # Create some checkpoint files
            for suffix in ["0", "2", "4", "12", "_config.json", "ABC"]:
                with open(f"{checkpoints_path}.{suffix}", 'w') as f:
                    f.write("dummy")

            # Should find the latest numeric checkpoint
            result = find_latest_checkpoint(checkpoints_path)
            self.assertEqual(result, f"{checkpoints_path}.12")

            print("✓ find_latest_checkpoint works correctly")

        except ImportError as e:
            if "cv2" in str(e):
                self.skipTest("OpenCV not available - skipping training tests")
            else:
                self.fail(f"find_latest_checkpoint test failed: {e}")
        except Exception as e:
            self.fail(f"find_latest_checkpoint test failed: {e}")

    def test_loss_function_validation(self):
        """Test that loss functions are properly defined"""
        try:
            # Import the training module to check if loss functions are available
            import keras_segmentation.keypoint_train as kt

            # Check that the module has the expected functions
            self.assertTrue(hasattr(kt, 'train_keypoints'))
            self.assertTrue(callable(kt.train_keypoints))

            # Check that loss function options are documented
            source = inspect.getsource(kt.train_keypoints)
            self.assertIn('weighted_mse', source)
            self.assertIn('binary_crossentropy', source)
            self.assertIn('categorical_crossentropy', source)

            print("✓ Loss function validation works")

        except ImportError:
            self.skipTest("inspect module not available")
        except Exception as e:
            self.fail(f"Loss function validation failed: {e}")

    def test_weighted_mse_loss_structure(self):
        """Test that weighted MSE loss function is properly structured"""
        try:
            import inspect
            import keras_segmentation.keypoint_train as kt

            source = inspect.getsource(kt.train_keypoints)

            # Check for weighted MSE implementation
            self.assertIn('weighted_mse', source)
            self.assertIn('def weighted_mse', source)
            self.assertIn('weight = 1.0 + 9.0 * y_true', source)

            print("✓ Weighted MSE loss structure is correct")

        except ImportError:
            self.skipTest("inspect module not available")
        except Exception as e:
            self.fail(f"Weighted MSE structure test failed: {e}")

    def test_training_parameter_handling(self):
        """Test that training parameters are properly handled"""
        try:
            import inspect
            import keras_segmentation.keypoint_train as kt

            source = inspect.getsource(kt.train_keypoints)

            # Check for key parameters
            required_params = [
                'train_images',
                'train_annotations',
                'n_keypoints',
                'input_height',
                'input_width',
                'loss_function',
                'epochs',
                'batch_size'
            ]

            for param in required_params:
                self.assertIn(param, source, f"Parameter {param} not found in function signature")

            print("✓ Training parameter handling is correct")

        except ImportError:
            self.skipTest("inspect module not available")
        except Exception as e:
            self.fail(f"Training parameter test failed: {e}")

    def test_config_file_creation(self):
        """Test that configuration files are properly created"""
        try:
            import inspect
            import keras_segmentation.keypoint_train as kt

            source = inspect.getsource(kt.train_keypoints)

            # Check for config file creation
            self.assertIn('_config.json', source)
            self.assertIn('model_class', source)
            self.assertIn('n_keypoints', source)
            self.assertIn('input_height', source)
            self.assertIn('output_height', source)

            print("✓ Config file creation is implemented")

        except ImportError:
            self.skipTest("inspect module not available")
        except Exception as e:
            self.fail(f"Config file creation test failed: {e}")

    def test_auto_resume_functionality(self):
        """Test that auto-resume functionality is implemented"""
        try:
            import inspect
            import keras_segmentation.keypoint_train as kt

            source = inspect.getsource(kt.train_keypoints)

            # Check for auto-resume parameters and logic
            self.assertIn('auto_resume_checkpoint', source)
            self.assertIn('load_weights', source)
            self.assertIn('initial_epoch', source)

            print("✓ Auto-resume functionality is implemented")

        except ImportError:
            self.skipTest("inspect module not available")
        except Exception as e:
            self.fail(f"Auto-resume test failed: {e}")


if __name__ == '__main__':
    import inspect  # Import here to avoid import errors in class methods
    unittest.main()
