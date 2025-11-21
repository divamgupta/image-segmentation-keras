#!/usr/bin/env python
"""
Example script showing how to use keypoint regression with keras-segmentation

This example demonstrates:
1. Creating synthetic keypoint data
2. Training a keypoint regression model
3. Making predictions and extracting keypoint coordinates
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras_segmentation.models.keypoint_models import keypoint_unet_mini
from keras_segmentation.keypoint_predict import predict_keypoint_coordinates

def create_synthetic_keypoint_data(num_samples=100, image_size=(224, 224), n_keypoints=5):
    """
    Create synthetic data for keypoint regression training

    Args:
        num_samples: Number of training samples to generate
        image_size: (height, width) of images
        n_keypoints: Number of keypoints to predict

    Returns:
        Saves images and heatmaps to train_images/ and train_keypoints/ directories
    """
    os.makedirs("train_images", exist_ok=True)
    os.makedirs("train_keypoints", exist_ok=True)
    os.makedirs("val_images", exist_ok=True)
    os.makedirs("val_keypoints", exist_ok=True)

    height, width = image_size

    for i in range(num_samples):
        # Create a blank image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Generate random keypoints
        keypoints = []
        heatmap = np.zeros((height, width, n_keypoints), dtype=np.float32)

        for k in range(n_keypoints):
            # Random keypoint position
            x = np.random.randint(20, width-20)
            y = np.random.randint(20, height-20)
            keypoints.append((x, y))

            # Create Gaussian heatmap around keypoint
            sigma = 10  # Gaussian spread
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
            heatmap[:, :, k] = gaussian

        # Draw keypoints on image for visualization
        for k, (x, y) in enumerate(keypoints):
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][k % 5]
            cv2.circle(img, (x, y), 3, color, -1)

        # Save image and heatmap
        if i < num_samples * 0.8:  # 80% training, 20% validation
            cv2.imwrite(f"train_images/sample_{i:03d}.png", img)
            np.save(f"train_keypoints/sample_{i:03d}.npy", heatmap)
        else:
            cv2.imwrite(f"val_images/sample_{i:03d}.png", img)
            np.save(f"val_keypoints/sample_{i:03d}.npy", heatmap)

    print(f"Created {num_samples} synthetic samples")
    print(f"Training samples: {int(num_samples * 0.8)}")
    print(f"Validation samples: {num_samples - int(num_samples * 0.8)}")


def train_keypoint_model():
    """Train a keypoint regression model"""
    from keras_segmentation.keypoint_train import train_keypoints

    print("Training keypoint regression model...")

    model = keypoint_unet_mini(
        n_keypoints=5,
        input_height=224,
        input_width=224
    )

    # Train the model
    model.train_keypoints(
        train_images="train_images/",
        train_annotations="train_keypoints/",
        input_height=224,
        input_width=224,
        n_keypoints=5,
        verify_dataset=False,  # Skip verification for synthetic data
        checkpoints_path="keypoint_checkpoints",
        epochs=10,
        batch_size=4,
        validate=True,
        val_images="val_images/",
        val_annotations="val_keypoints/",
        val_batch_size=4,
        auto_resume_checkpoint=False,
        loss_function='weighted_mse',  # Use weighted MSE for better keypoint detection
        steps_per_epoch=20,
        val_steps_per_epoch=5
    )

    print("Training completed!")
    return model


def test_keypoint_prediction(model):
    """Test keypoint prediction on a sample image"""
    print("Testing keypoint prediction...")

    # Load a test image
    test_img_path = "val_images/sample_080.png"
    if not os.path.exists(test_img_path):
        print("Test image not found, creating a simple test...")
        # Create a simple test image
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add some keypoints manually
        keypoints = [(50, 50), (100, 100), (150, 150), (200, 50), (50, 200)]
        for k, (x, y) in enumerate(keypoints):
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][k]
            cv2.circle(img, (x, y), 3, color, -1)
        cv2.imwrite("test_image.png", img)
        test_img_path = "test_image.png"

    # Make prediction
    heatmap = model.predict_keypoints(inp=test_img_path, out_fname="prediction")

    print(f"Heatmap shape: {heatmap.shape}")

    # Extract keypoint coordinates
    all_keypoints = []
    for k in range(heatmap.shape[2]):
        keypoints = predict_keypoint_coordinates(heatmap[:, :, k], threshold=0.1)
        all_keypoints.append(keypoints)
        print(f"Keypoint {k}: {keypoints}")

    # Visualize results
    visualize_prediction(test_img_path, heatmap, all_keypoints)


def visualize_prediction(image_path, heatmap, keypoints):
    """Visualize the prediction results"""
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Heatmap overlay
    plt.subplot(1, 3, 2)
    heatmap_max = np.max(heatmap, axis=2)
    plt.imshow(img, alpha=0.7)
    plt.imshow(heatmap_max, alpha=0.3, cmap='hot')
    plt.title("Heatmap Overlay")
    plt.axis('off')

    # Predicted keypoints
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    colors = ['red', 'green', 'blue', 'yellow', 'magenta']
    for k, kp_list in enumerate(keypoints):
        for x, y, conf in kp_list:
            plt.scatter(x, y, c=colors[k % len(colors)], s=50, alpha=0.8)
            plt.text(x+5, y+5, '.1f', fontsize=8, color=colors[k % len(colors)])
    plt.title("Predicted Keypoints")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("keypoint_prediction_result.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("Results saved to 'keypoint_prediction_result.png'")


def main():
    """Main function to run the complete keypoint regression example"""
    print("=" * 60)
    print("Keypoint Regression Example with keras-segmentation")
    print("=" * 60)

    # Step 1: Create synthetic data
    print("\nStep 1: Creating synthetic keypoint data...")
    create_synthetic_keypoint_data(num_samples=100, n_keypoints=5)

    # Step 2: Train model
    print("\nStep 2: Training keypoint regression model...")
    try:
        model = train_keypoint_model()
    except Exception as e:
        print(f"Training failed: {e}")
        print("Trying to load existing model...")
        model = keypoint_unet_mini(n_keypoints=5, input_height=224, input_width=224)
        # Try to load weights if they exist
        try:
            model.load_weights("keypoint_checkpoints.0009")  # Load last checkpoint
            print("Loaded existing model weights")
        except:
            print("No existing model found. Please run training first.")
            return

    # Step 3: Test prediction
    print("\nStep 3: Testing keypoint prediction...")
    test_keypoint_prediction(model)

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("Check the generated files:")
    print("- keypoint_prediction_result.png: Visualization of results")
    print("- prediction_keypoint_*.png: Individual keypoint heatmaps")
    print("=" * 60)


if __name__ == "__main__":
    main()
