import unittest
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

class TestKeypointModels(unittest.TestCase):
    """Test keypoint regression models"""

    def test_keypoint_unet_mini_creation(self):
        """Test that keypoint_unet_mini can be created"""
        try:
            from keras_segmentation.models.keypoint_models import keypoint_unet_mini
            model = keypoint_unet_mini(n_keypoints=5, input_height=224, input_width=224)
            self.assertIsNotNone(model)
            self.assertEqual(model.n_keypoints, 5)
            self.assertEqual(model.input_height, 224)
            self.assertEqual(model.input_width, 224)
            print("✓ keypoint_unet_mini creation successful")
        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"keypoint_unet_mini creation failed: {e}")

    def test_keypoint_unet_creation(self):
        """Test that keypoint_unet can be created"""
        try:
            from keras_segmentation.models.keypoint_models import keypoint_unet
            model = keypoint_unet(n_keypoints=17, input_height=224, input_width=224)
            self.assertIsNotNone(model)
            self.assertEqual(model.n_keypoints, 17)
            print("✓ keypoint_unet creation successful")
        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"keypoint_unet creation failed: {e}")

    def test_keypoint_vgg_unet_creation(self):
        """Test that keypoint_vgg_unet can be created"""
        try:
            from keras_segmentation.models.keypoint_models import keypoint_vgg_unet
            model = keypoint_vgg_unet(n_keypoints=17, input_height=224, input_width=224)
            self.assertIsNotNone(model)
            self.assertEqual(model.n_keypoints, 17)
            print("✓ keypoint_vgg_unet creation successful")
        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"keypoint_vgg_unet creation failed: {e}")

    def test_model_registry_includes_keypoint_models(self):
        """Test that keypoint models are registered in model_from_name"""
        try:
            from keras_segmentation.models.all_models import model_from_name

            keypoint_models = [
                'keypoint_unet_mini',
                'keypoint_unet',
                'keypoint_vgg_unet',
                'keypoint_resnet50_unet',
                'keypoint_mobilenet_unet'
            ]

            for model_name in keypoint_models:
                self.assertIn(model_name, model_from_name,
                            f"Model {model_name} not found in registry")
                # Verify the function is callable
                self.assertTrue(callable(model_from_name[model_name]),
                              f"Model {model_name} is not callable")

            print("✓ All keypoint models registered successfully")

        except Exception as e:
            self.fail(f"Model registry test failed: {e}")


if __name__ == '__main__':
    unittest.main()

