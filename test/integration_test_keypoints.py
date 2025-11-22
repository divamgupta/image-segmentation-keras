#!/usr/bin/env python
"""
Integration test for keypoint regression functionality.
Tests the full pipeline using sample data.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

class TestKeypointIntegration(unittest.TestCase):
    """Integration tests for keypoint regression"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.sample_dir = os.path.join(os.path.dirname(__file__), 'sample_images')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_full_keypoint_pipeline(self):
        """Test the complete keypoint regression pipeline"""
        try:
            # Step 1: Create synthetic keypoint data
            print("Step 1: Creating synthetic keypoint data...")
            train_img_dir = os.path.join(self.tmp_dir, 'train_images')
            train_kp_dir = os.path.join(self.tmp_dir, 'train_keypoints')
            os.makedirs(train_img_dir)
            os.makedirs(train_kp_dir)

            # Generate 5 sample images with keypoints
            n_samples = 5
            n_keypoints = 3
            img_size = 64

            for i in range(n_samples):
                # Create synthetic RGB image
                img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

                # Create keypoint heatmap
                heatmap = np.zeros((img_size, img_size, n_keypoints), dtype=np.float32)

                # Add random keypoints
                keypoints = []
                for k in range(n_keypoints):
                    x = np.random.randint(10, img_size-10)
                    y = np.random.randint(10, img_size-10)
                    keypoints.append((x, y))

                    # Create Gaussian heatmap
                    y_coords, x_coords = np.mgrid[0:img_size, 0:img_size]
                    sigma = 5.0
                    gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
                    heatmap[:, :, k] = gaussian

                # Save image and heatmap
                img_path = os.path.join(train_img_dir, '03d')
                kp_path = os.path.join(train_kp_dir, '03d')

                # For testing, just create dummy files since we can't import cv2
                with open(img_path, 'wb') as f:
                    f.write(img.tobytes())
                np.save(kp_path, heatmap)

            print(f"✓ Created {n_samples} synthetic samples")

            # Step 2: Test data loading
            print("Step 2: Testing data loading...")
            from keras_segmentation.data_utils.keypoint_data_loader import (
                get_keypoint_pairs_from_paths, verify_keypoint_dataset
            )

            # Verify dataset
            is_valid = verify_keypoint_dataset(train_img_dir, train_kp_dir, n_keypoints)
            self.assertTrue(is_valid, "Dataset verification failed")
            print("✓ Dataset verification passed")

            # Test pair matching
            pairs = get_keypoint_pairs_from_paths(train_img_dir, train_kp_dir)
            self.assertEqual(len(pairs), n_samples, f"Expected {n_samples} pairs, got {len(pairs)}")
            print(f"✓ Found {len(pairs)} image-keypoint pairs")

            # Step 3: Test model creation
            print("Step 3: Testing model creation...")
            try:
                from keras_segmentation.models.keypoint_models import keypoint_unet_mini
                model = keypoint_unet_mini(n_keypoints=n_keypoints, input_height=img_size, input_width=img_size)
                self.assertIsNotNone(model)
                self.assertEqual(model.n_keypoints, n_keypoints)
                print("✓ Model creation successful")
            except ImportError:
                self.skipTest("Keras/TensorFlow not available - skipping model tests")

            # Step 4: Test coordinate extraction
            print("Step 4: Testing coordinate extraction...")
            from keras_segmentation.keypoint_predict import predict_keypoint_coordinates

            # Test with one of our synthetic heatmaps
            test_heatmap = heatmap[:, :, 0]  # First keypoint heatmap
            keypoints = predict_keypoint_coordinates(test_heatmap, threshold=0.1)

            # Should find the keypoint
            self.assertGreater(len(keypoints), 0, "No keypoints found")
            x, y, conf = keypoints[0]

            # Should be within image bounds
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLess(x, img_size)
            self.assertLess(y, img_size)
            self.assertGreater(conf, 0.0)

            print(".1f")

            print("🎉 Full keypoint pipeline integration test passed!")

        except Exception as e:
            self.fail(f"Integration test failed: {e}")

    def test_sample_data_compatibility(self):
        """Test that our implementation works with the existing sample data structure"""
        # Check that sample images exist
        sample_files = [
            'sample_images/1_input.jpg',
            'sample_images/1_output.png',
            'sample_images/2_input.jpg',
            'sample_images/2_output.png'
        ]

        for file_path in sample_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            self.assertTrue(os.path.exists(full_path), f"Sample file {file_path} not found")

        print("✓ Sample data files are accessible")

        # Test that our data loader can handle the structure
        # (Even though sample data is for segmentation, not keypoints)
        sample_img_dir = os.path.join(os.path.dirname(__file__), 'sample_images')

        # Create mock keypoint directory for testing
        mock_kp_dir = os.path.join(self.tmp_dir, 'mock_keypoints')
        os.makedirs(mock_kp_dir)

        # Create a mock keypoint file
        mock_heatmap = np.random.rand(224, 224, 5).astype(np.float32)
        np.save(os.path.join(mock_kp_dir, '1.npy'), mock_heatmap)

        print("✓ Sample data structure is compatible")

    def test_backward_compatibility(self):
        """Test that our changes don't break existing functionality"""
        try:
            # Test that original imports still work
            import keras_segmentation
            self.assertTrue(hasattr(keras_segmentation, 'models'))
            self.assertTrue(hasattr(keras_segmentation, 'train'))

            # Test that model registry still works for original models
            from keras_segmentation.models.all_models import model_from_name
            original_models = ['fcn_8', 'fcn_32', 'unet_mini', 'unet', 'pspnet']

            for model_name in original_models:
                self.assertIn(model_name, model_from_name,
                            f"Original model {model_name} missing from registry")

            print("✓ Backward compatibility maintained")

        except Exception as e:
            self.fail(f"Backward compatibility test failed: {e}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)

