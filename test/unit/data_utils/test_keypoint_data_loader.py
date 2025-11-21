import unittest
import numpy as np
import tempfile
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

class TestKeypointDataLoader(unittest.TestCase):
    """Test keypoint data loading functionality"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir)

    def test_get_keypoint_array_from_numpy(self):
        """Test loading keypoint array from numpy array"""
        try:
            from keras_segmentation.data_utils.keypoint_data_loader import get_keypoint_array

            # Create a synthetic heatmap
            height, width, n_keypoints = 64, 64, 3
            heatmap = np.zeros((height, width, n_keypoints), dtype=np.float32)

            # Add keypoints at specific locations
            keypoints = [(20, 20), (40, 40), (50, 30)]
            sigma = 3.0

            for i, (x, y) in enumerate(keypoints):
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
                heatmap[:, :, i] = gaussian

            # Test the function
            result = get_keypoint_array(heatmap, n_keypoints, width, height)

            # Verify shape
            self.assertEqual(result.shape, (height * width, n_keypoints))

            # Verify data type
            self.assertEqual(result.dtype, np.float32)

            # Verify value range
            self.assertTrue(np.min(result) >= 0.0)
            self.assertTrue(np.max(result) <= 1.0)

            print("✓ get_keypoint_array from numpy array works correctly")

        except Exception as e:
            self.fail(f"get_keypoint_array test failed: {e}")

    def test_get_keypoint_array_from_file(self):
        """Test loading keypoint array from .npy file"""
        try:
            from keras_segmentation.data_utils.keypoint_data_loader import get_keypoint_array

            # Create a synthetic heatmap
            height, width, n_keypoints = 32, 32, 2
            heatmap = np.random.rand(height, width, n_keypoints).astype(np.float32)

            # Save to file
            npy_path = os.path.join(self.tmp_dir, 'test_keypoints.npy')
            np.save(npy_path, heatmap)

            # Load using the function
            result = get_keypoint_array(npy_path, n_keypoints, width, height)

            # Verify shape
            self.assertEqual(result.shape, (height * width, n_keypoints))

            # Verify values are close (may be resized)
            self.assertTrue(np.allclose(result.reshape(height, width, n_keypoints), heatmap, rtol=0.1))

            print("✓ get_keypoint_array from .npy file works correctly")

        except Exception as e:
            self.fail(f"get_keypoint_array from file test failed: {e}")

    def test_verify_keypoint_dataset(self):
        """Test keypoint dataset verification"""
        try:
            from keras_segmentation.data_utils.keypoint_data_loader import verify_keypoint_dataset

            # Create mock image and keypoint files
            img_dir = os.path.join(self.tmp_dir, 'images')
            kp_dir = os.path.join(self.tmp_dir, 'keypoints')
            os.makedirs(img_dir)
            os.makedirs(kp_dir)

            # Create a mock image file
            img_path = os.path.join(img_dir, 'test_001.jpg')
            mock_img = np.zeros((64, 64, 3), dtype=np.uint8)
            mock_img.tofile(img_path)  # Create a dummy file

            # Create a corresponding keypoint file
            kp_path = os.path.join(kp_dir, 'test_001.npy')
            heatmap = np.random.rand(64, 64, 5).astype(np.float32)
            np.save(kp_path, heatmap)

            # Test verification
            result = verify_keypoint_dataset(img_dir, kp_dir, n_keypoints=5)

            # Should pass (basic verification)
            self.assertTrue(result)

            print("✓ verify_keypoint_dataset works correctly")

        except Exception as e:
            self.fail(f"verify_keypoint_dataset test failed: {e}")

    def test_keypoint_generator_basic(self):
        """Test basic keypoint generator functionality"""
        try:
            from keras_segmentation.data_utils.keypoint_data_loader import keypoint_generator

            # Create mock directories and files
            img_dir = os.path.join(self.tmp_dir, 'images')
            kp_dir = os.path.join(self.tmp_dir, 'keypoints')
            os.makedirs(img_dir)
            os.makedirs(kp_dir)

            # Create test files
            for i in range(3):
                # Mock image
                img_path = os.path.join(img_dir, f'test_{i:03d}.jpg')
                with open(img_path, 'wb') as f:
                    f.write(b'dummy_image_data')

                # Keypoint heatmap
                kp_path = os.path.join(kp_dir, f'test_{i:03d}.npy')
                heatmap = np.random.rand(32, 32, 5).astype(np.float32)
                np.save(kp_path, heatmap)

            # Test generator
            gen = keypoint_generator(
                img_dir, kp_dir, batch_size=2, n_keypoints=5,
                input_height=32, input_width=32,
                output_height=32, output_width=32
            )

            # Get one batch
            X_batch, Y_batch = next(gen)

            # Verify batch structure
            self.assertEqual(len(X_batch), 2)  # batch_size
            self.assertEqual(len(Y_batch), 2)  # batch_size

            print("✓ keypoint_generator works correctly")

        except ImportError:
            self.skipTest("OpenCV not available for image loading")
        except Exception as e:
            self.fail(f"keypoint_generator test failed: {e}")


if __name__ == '__main__':
    unittest.main()

