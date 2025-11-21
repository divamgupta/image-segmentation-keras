import unittest
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

class TestKeypointPrediction(unittest.TestCase):
    """Test keypoint prediction functionality"""

    def test_predict_keypoint_coordinates_perfect_gaussian(self):
        """Test coordinate extraction with perfect Gaussian heatmap"""
        try:
            from keras_segmentation.keypoint_predict import predict_keypoint_coordinates

            # Create a perfect Gaussian centered at (50, 50)
            heatmap = np.zeros((100, 100), dtype=np.float32)
            y_coords, x_coords = np.mgrid[0:100, 0:100]
            sigma = 5.0
            gaussian = np.exp(-((x_coords - 50)**2 + (y_coords - 50)**2) / (2 * sigma**2))
            heatmap = gaussian / np.max(gaussian)

            # Extract coordinates
            keypoints = predict_keypoint_coordinates(heatmap, threshold=0.1)

            # Should find exactly one keypoint
            self.assertEqual(len(keypoints), 1)

            x, y, conf = keypoints[0]

            # Should be very close to the center (within 0.1 pixels)
            self.assertAlmostEqual(x, 50.0, delta=0.1)
            self.assertAlmostEqual(y, 50.0, delta=0.1)
            self.assertGreater(conf, 0.9)  # High confidence

            print("✓ Perfect Gaussian coordinate extraction works")

        except ImportError as e:
            if "cv2" in str(e):
                self.skipTest("OpenCV not available - skipping coordinate extraction tests")
            else:
                self.fail(f"Perfect Gaussian test failed: {e}")
        except ImportError as e:
            if "cv2" in str(e):
                self.skipTest("OpenCV not available - skipping coordinate extraction tests")
            else:
                self.fail(f"Perfect Gaussian test failed: {e}")
        except Exception as e:
            self.fail(f"Perfect Gaussian test failed: {e}")

    def test_predict_keypoint_coordinates_offset_gaussian(self):
        """Test coordinate extraction with offset Gaussian"""
        try:
            from keras_segmentation.keypoint_predict import predict_keypoint_coordinates

            # Create Gaussian at (75.3, 42.7) - non-integer coordinates
            center_x, center_y = 75.3, 42.7
            heatmap = np.zeros((100, 100), dtype=np.float32)
            y_coords, x_coords = np.mgrid[0:100, 0:100]
            sigma = 8.0
            gaussian = np.exp(-((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * sigma**2))
            heatmap = gaussian / np.max(gaussian)

            # Extract coordinates
            keypoints = predict_keypoint_coordinates(heatmap, threshold=0.05)

            self.assertEqual(len(keypoints), 1)
            x, y, conf = keypoints[0]

            # Should be very close to the actual center
            self.assertAlmostEqual(x, center_x, delta=0.2)
            self.assertAlmostEqual(y, center_y, delta=0.2)

            print("✓ Offset Gaussian coordinate extraction works")

        except Exception as e:
            self.fail(f"Offset Gaussian test failed: {e}")

    def test_predict_keypoint_coordinates_threshold_filtering(self):
        """Test that low-confidence keypoints are filtered out"""
        try:
            from keras_segmentation.keypoint_predict import predict_keypoint_coordinates

            # Create two Gaussians - one strong, one weak
            heatmap = np.zeros((100, 100), dtype=np.float32)
            y_coords, x_coords = np.mgrid[0:100, 0:100]
            sigma = 3.0

            # Strong keypoint
            strong_gaussian = np.exp(-((x_coords - 30)**2 + (y_coords - 30)**2) / (2 * sigma**2))
            heatmap += strong_gaussian

            # Weak keypoint (much smaller amplitude)
            weak_gaussian = 0.05 * np.exp(-((x_coords - 70)**2 + (y_coords - 70)**2) / (2 * sigma**2))
            heatmap += weak_gaussian

            # Extract with high threshold
            keypoints = predict_keypoint_coordinates(heatmap, threshold=0.5)

            # Should only find the strong keypoint
            self.assertEqual(len(keypoints), 1)

            x, y, conf = keypoints[0]
            self.assertAlmostEqual(x, 30.0, delta=1.0)
            self.assertAlmostEqual(y, 30.0, delta=1.0)
            self.assertGreater(conf, 0.8)

            print("✓ Threshold filtering works correctly")

        except ImportError as e:
            if "cv2" in str(e):
                self.skipTest("OpenCV not available - skipping coordinate extraction tests")
            else:
                self.fail(f"Threshold filtering test failed: {e}")
        except Exception as e:
            self.fail(f"Threshold filtering test failed: {e}")

    def test_predict_keypoint_coordinates_multiple_peaks(self):
        """Test coordinate extraction with multiple peaks"""
        try:
            from keras_segmentation.keypoint_predict import predict_keypoint_coordinates

            # Create heatmap with two distinct peaks
            heatmap = np.zeros((100, 100), dtype=np.float32)
            y_coords, x_coords = np.mgrid[0:100, 0:100]
            sigma = 4.0

            # First peak
            gaussian1 = np.exp(-((x_coords - 25)**2 + (y_coords - 25)**2) / (2 * sigma**2))
            heatmap += gaussian1

            # Second peak
            gaussian2 = np.exp(-((x_coords - 75)**2 + (y_coords - 75)**2) / (2 * sigma**2))
            heatmap += gaussian2

            # Normalize
            heatmap = heatmap / np.max(heatmap)

            # Extract coordinates
            keypoints = predict_keypoint_coordinates(heatmap, threshold=0.1, max_peaks=2)

            # Should find both keypoints
            self.assertEqual(len(keypoints), 2)

            # Sort by x coordinate
            keypoints.sort(key=lambda k: k[0])

            # Check first keypoint
            x1, y1, conf1 = keypoints[0]
            self.assertAlmostEqual(x1, 25.0, delta=1.0)
            self.assertAlmostEqual(y1, 25.0, delta=1.0)

            # Check second keypoint
            x2, y2, conf2 = keypoints[1]
            self.assertAlmostEqual(x2, 75.0, delta=1.0)
            self.assertAlmostEqual(y2, 75.0, delta=1.0)

            print("✓ Multiple peaks detection works correctly")

        except ImportError as e:
            if "cv2" in str(e):
                self.skipTest("OpenCV not available - skipping coordinate extraction tests")
            else:
                self.fail(f"Multiple peaks test failed: {e}")
        except Exception as e:
            self.fail(f"Multiple peaks test failed: {e}")

    def test_predict_keypoint_coordinates_no_peaks(self):
        """Test behavior with no peaks above threshold"""
        try:
            from keras_segmentation.keypoint_predict import predict_keypoint_coordinates

            # Create a very flat heatmap (no clear peaks)
            heatmap = np.ones((50, 50), dtype=np.float32) * 0.05  # Low uniform values

            # Extract coordinates
            keypoints = predict_keypoint_coordinates(heatmap, threshold=0.1)

            # Should find no keypoints
            self.assertEqual(len(keypoints), 0)

            print("✓ No peaks detection works correctly")

        except ImportError as e:
            if "cv2" in str(e):
                self.skipTest("OpenCV not available - skipping coordinate extraction tests")
            else:
                self.fail(f"No peaks test failed: {e}")
        except Exception as e:
            self.fail(f"No peaks test failed: {e}")

    def test_weighted_average_accuracy(self):
        """Test that weighted average gives sub-pixel accuracy"""
        try:
            from keras_segmentation.keypoint_predict import predict_keypoint_coordinates

            # Create asymmetric heatmap to test weighted average
            heatmap = np.zeros((20, 20), dtype=np.float32)

            # Create a keypoint that's not at a pixel center
            center_x, center_y = 10.7, 8.3

            y_coords, x_coords = np.mgrid[0:20, 0:20]
            sigma = 2.5
            gaussian = np.exp(-((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * sigma**2))
            heatmap = gaussian / np.max(gaussian)

            # Extract coordinates
            keypoints = predict_keypoint_coordinates(heatmap, threshold=0.1)

            self.assertEqual(len(keypoints), 1)
            x, y, conf = keypoints[0]

            # Should be very close to the true center (within 0.1 pixels)
            self.assertAlmostEqual(x, center_x, delta=0.1)
            self.assertAlmostEqual(y, center_y, delta=0.1)

            print(".3f")

        except ImportError as e:
            if "cv2" in str(e):
                self.skipTest("OpenCV not available - skipping coordinate extraction tests")
            else:
                self.fail(f"Weighted average accuracy test failed: {e}")
        except Exception as e:
            self.fail(f"Weighted average accuracy test failed: {e}")


if __name__ == '__main__':
    unittest.main()
