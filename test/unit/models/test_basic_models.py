import unittest
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

class TestBasicModels(unittest.TestCase):
    """Test basic model utilities and the reshape fix"""

    def test_segmentation_model_reshape_fix(self):
        """Test that the Reshape+Permute fix works correctly for both channel orderings"""
        try:
            from keras_segmentation.models.model_utils import get_segmentation_model
            from keras_segmentation.models.config import IMAGE_ORDERING
            from keras.layers import Input, Conv2D

            # Test with both channel orderings
            for test_ordering in ['channels_first', 'channels_last']:
                with self.subTest(channel_ordering=test_ordering):
                    # Temporarily set the channel ordering
                    import keras_segmentation.models.config as config_module
                    original_ordering = config_module.IMAGE_ORDERING
                    config_module.IMAGE_ORDERING = test_ordering

                    try:
                        # Create test parameters
                        input_height, input_width, n_classes = 32, 32, 3
                        batch_size = 2

                        # Create input tensor based on channel ordering
                        if test_ordering == 'channels_first':
                            input_shape = (n_classes, input_height, input_width)
                        else:
                            input_shape = (input_height, input_width, n_classes)

                        img_input = Input(shape=input_shape, batch_size=batch_size)

                        # Create a simple conv layer as segmentation output
                        o = Conv2D(n_classes, (1, 1), padding='same')(img_input)

                        # Get the segmentation model (this applies the reshape operations)
                        model = get_segmentation_model(img_input, o)

                        # Verify output shape is correct: (batch, height*width, n_classes)
                        expected_shape = (batch_size, input_height * input_width, n_classes)
                        self.assertEqual(model.output_shape, expected_shape,
                                       f"Failed for {test_ordering}: expected {expected_shape}, got {model.output_shape}")

                        # Test with dummy data to ensure it actually works
                        dummy_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
                        prediction = model.predict(dummy_input, verbose=0)

                        self.assertEqual(prediction.shape, expected_shape,
                                       f"Prediction shape failed for {test_ordering}: expected {expected_shape}, got {prediction.shape}")

                        print(f"✓ Reshape fix works correctly for {test_ordering}")

                    finally:
                        # Restore original ordering
                        config_module.IMAGE_ORDERING = original_ordering

        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"Reshape fix test failed: {e}")

    def test_vanilla_encoder_import(self):
        """Test that vanilla_encoder can be imported"""
        try:
            from keras_segmentation.models.basic_models import vanilla_encoder
            self.assertTrue(callable(vanilla_encoder))
            print("✓ vanilla_encoder import successful")
        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"vanilla_encoder import failed: {e}")

    def test_vanilla_encoder_default_params(self):
        """Test vanilla_encoder with default parameters"""
        try:
            from keras_segmentation.models.basic_models import vanilla_encoder
            img_input, levels = vanilla_encoder()

            # Check input tensor
            self.assertIsNotNone(img_input)
            self.assertEqual(len(img_input.shape), 4)  # [batch, height, width, channels] or [batch, channels, height, width]

            # Check levels list
            self.assertIsInstance(levels, list)
            self.assertEqual(len(levels), 5)  # Should have 5 encoder levels

            # Check that all levels are tensors
            for i, level in enumerate(levels):
                self.assertIsNotNone(level)
                print(f"✓ Level {i+1} created successfully")

            print("✓ vanilla_encoder with default params successful")

        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"vanilla_encoder default params test failed: {e}")

    def test_vanilla_encoder_custom_dimensions(self):
        """Test vanilla_encoder with custom input dimensions"""
        try:
            from keras_segmentation.models.basic_models import vanilla_encoder

            test_cases = [
                (128, 128, 3),   # Smaller square
                (256, 128, 3),   # Rectangular
                (224, 224, 1),   # Grayscale
                (320, 240, 4),   # RGBA
            ]

            for height, width, channels in test_cases:
                with self.subTest(height=height, width=width, channels=channels):
                    img_input, levels = vanilla_encoder(
                        input_height=height,
                        input_width=width,
                        channels=channels
                    )

                    # Verify input shape based on IMAGE_ORDERING
                    from keras_segmentation.models.config import IMAGE_ORDERING
                    if IMAGE_ORDERING == 'channels_last':
                        expected_shape = (height, width, channels)
                    else:  # channels_first
                        expected_shape = (channels, height, width)

                    # Check that the last 3 dimensions match (excluding batch dimension)
                    self.assertEqual(img_input.shape[1:], expected_shape)

                    # Verify levels exist
                    self.assertEqual(len(levels), 5)

                    print(f"✓ Custom dimensions ({height}x{width}x{channels}) successful")

        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"vanilla_encoder custom dimensions test failed: {e}")

    def test_vanilla_encoder_output_shapes(self):
        """Test that vanilla_encoder produces expected output shapes"""
        try:
            from keras_segmentation.models.basic_models import vanilla_encoder
            from keras_segmentation.models.config import IMAGE_ORDERING

            img_input, levels = vanilla_encoder(input_height=224, input_width=224, channels=3)

            # Expected spatial dimensions after each pooling operation
            # Input: 224x224 -> Level 0: 112x112, Level 1: 56x56, Levels 2-4: 28x28 each
            expected_spatial_dims = [(112, 112), (56, 56), (28, 28), (28, 28), (28, 28)]
            expected_channels = [64, 128, 256, 256, 256]

            for i, (level, expected_dim, expected_chan) in enumerate(zip(levels, expected_spatial_dims, expected_channels)):
                # Check spatial dimensions (should be consistent regardless of channel ordering)
                if IMAGE_ORDERING == 'channels_last':
                    self.assertEqual(level.shape[1:3], expected_dim, f"Level {i} spatial dims incorrect")
                    self.assertEqual(level.shape[3], expected_chan, f"Level {i} channels incorrect")
                else:  # channels_first
                    self.assertEqual(level.shape[2:4], expected_dim, f"Level {i} spatial dims incorrect")
                    self.assertEqual(level.shape[1], expected_chan, f"Level {i} channels incorrect")

                print(f"✓ Level {i} shape validation successful")

        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"vanilla_encoder output shapes test failed: {e}")

    def test_vanilla_encoder_tensor_types(self):
        """Test that vanilla_encoder returns proper Keras tensors"""
        try:
            from keras_segmentation.models.basic_models import vanilla_encoder
            from keras.engine.keras_tensor import KerasTensor

            img_input, levels = vanilla_encoder()

            # Check input tensor type
            self.assertIsInstance(img_input, KerasTensor)

            # Check all level tensors
            for i, level in enumerate(levels):
                self.assertIsInstance(level, KerasTensor, f"Level {i} is not a Keras tensor")

            print("✓ All tensors are proper Keras tensors")

        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"vanilla_encoder tensor types test failed: {e}")

    def test_vanilla_encoder_no_empty_levels(self):
        """Test that vanilla_encoder doesn't return empty levels"""
        try:
            from keras_segmentation.models.basic_models import vanilla_encoder

            img_input, levels = vanilla_encoder()

            # Ensure no level is None or empty
            for i, level in enumerate(levels):
                self.assertIsNotNone(level, f"Level {i} is None")
                self.assertGreater(np.prod(level.shape[1:]), 0, f"Level {i} has zero volume")

            print("✓ No empty levels in vanilla_encoder output")

        except ImportError:
            self.skipTest("Keras/TensorFlow not available")
        except Exception as e:
            self.fail(f"vanilla_encoder empty levels test failed: {e}")


if __name__ == '__main__':
    unittest.main()
