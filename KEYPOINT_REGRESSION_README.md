# Keypoint Regression with keras-segmentation

This document explains how to use the new keypoint regression functionality added to keras-segmentation, which allows you to predict keypoint heatmaps instead of semantic segmentation masks.

## Overview

The standard keras-segmentation library is designed for semantic segmentation where each pixel belongs to exactly one class. For keypoint regression, we need:

- **Independent probabilities**: Each keypoint should have a probability map from 0-1, independent of other keypoints
- **Continuous predictions**: Instead of discrete class labels, we predict continuous heatmap values
- **Flexible loss functions**: Support for regression losses like MSE instead of categorical cross-entropy

## Key Differences from Segmentation

| Aspect | Segmentation | Keypoint Regression |
|--------|-------------|-------------------|
| Output Activation | Softmax (sums to 1) | Sigmoid (independent 0-1) |
| Loss Function | Categorical Cross-entropy | MSE, Binary Cross-entropy, Weighted MSE |
| Data Format | Integer class labels | Float32 heatmaps [0-1] |
| Training Method | `model.train()` | `model.train_keypoints()` |
| Prediction Method | `model.predict_segmentation()` | `model.predict_keypoints()` |

## Quick Start

```python
from keras_segmentation.models.keypoint_models import keypoint_unet_mini

# Create a keypoint regression model for 17 keypoints
model = keypoint_unet_mini(n_keypoints=17, input_height=224, input_width=224)

# Train the model
model.train_keypoints(
    train_images="path/to/images/",
    train_annotations="path/to/heatmaps/",
    n_keypoints=17,
    epochs=50,
    loss_function='weighted_mse'  # Better for sparse keypoints
)

# Make predictions
heatmap = model.predict_keypoints(inp="test_image.jpg")
```

## Available Models

The following keypoint regression models are available:

- `keypoint_unet_mini` - Lightweight U-Net for quick experimentation
- `keypoint_unet` - Standard U-Net architecture
- `keypoint_vgg_unet` - U-Net with VGG16 encoder
- `keypoint_resnet50_unet` - U-Net with ResNet50 encoder
- `keypoint_mobilenet_unet` - U-Net with MobileNet encoder (for mobile deployment)

## Data Format

### Images
Standard RGB images in JPG/PNG format, same as segmentation.

### Keypoint Annotations
Keypoint annotations should be provided as:

1. **NumPy arrays (.npy files)**: Shape `(height, width, n_keypoints)` with float32 values in [0, 1]
2. **PNG images**: Single channel for 1 keypoint, or RGB for up to 3 keypoints

**File naming**: Images and heatmaps must have matching filenames:
```
images/person_001.jpg -> heatmaps/person_001.npy
images/person_002.png -> heatmaps/person_002.png
```

### Creating Heatmaps

For each keypoint, create a 2D Gaussian heatmap centered at the keypoint location:

```python
import numpy as np

def create_heatmap(height, width, keypoints, sigma=10):
    """
    Create Gaussian heatmaps for keypoints

    Args:
        height, width: Image dimensions
        keypoints: List of (x, y) coordinates
        sigma: Gaussian standard deviation

    Returns:
        heatmap: (height, width, n_keypoints) float32 array
    """
    heatmap = np.zeros((height, width, len(keypoints)), dtype=np.float32)

    for i, (x, y) in enumerate(keypoints):
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
        heatmap[:, :, i] = gaussian

    return heatmap
```

## Training

### Basic Training

```python
from keras_segmentation.keypoint_train import train_keypoints

model.train_keypoints(
    train_images="train_images/",
    train_annotations="train_heatmaps/",
    input_height=224,
    input_width=224,
    n_keypoints=17,
    epochs=50,
    batch_size=8,
    validate=True,
    val_images="val_images/",
    val_annotations="val_heatmaps/",
    loss_function='weighted_mse',  # 'mse', 'binary_crossentropy', or 'weighted_mse'
    checkpoints_path="checkpoints",
    auto_resume_checkpoint=True
)
```

### Loss Functions

- **'mse'**: Standard mean squared error
- **'binary_crossentropy'**: Binary cross-entropy (treats each keypoint independently)
- **'weighted_mse'**: Weighted MSE that gives 10x weight to keypoint pixels vs background

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `loss_function` | Loss function to use | 'mse' |
| `steps_per_epoch` | Steps per training epoch | 512 |
| `optimizer_name` | Optimizer ('adam', 'sgd', etc.) | 'adam' |
| `verify_dataset` | Verify data integrity before training | True |

## Prediction

### Basic Prediction

```python
# Predict heatmaps
heatmap = model.predict_keypoints(inp="image.jpg")

# Save individual keypoint heatmaps
heatmap = model.predict_keypoints(inp="image.jpg", out_fname="prediction")

# Save as numpy array
heatmap = model.predict_keypoints(inp="image.jpg", keypoints_fname="keypoints.npy")
```

### Extract Keypoint Coordinates

```python
from keras_segmentation.keypoint_predict import predict_keypoint_coordinates

# For each keypoint channel
keypoints = []
for k in range(heatmap.shape[2]):
    kp_coords = predict_keypoint_coordinates(heatmap[:, :, k], threshold=0.1)
    keypoints.append(kp_coords)  # List of (x, y, confidence) tuples

# keypoints[k] contains detected coordinates for keypoint k
for i, kp_list in enumerate(keypoints):
    print(f"Keypoint {i}: {kp_list}")
```

### Coordinate Extraction Options

- `threshold`: Minimum confidence threshold (0-1)
- `max_peaks`: Maximum number of peaks to detect per keypoint

## Data Preparation Tools

### Synthetic Data Generation

Use the provided example script to generate synthetic keypoint data:

```bash
python example_keypoint_regression.py
```

This creates:
- Synthetic images with drawn keypoints
- Corresponding Gaussian heatmaps
- Training/validation splits

### Data Verification

```python
from keras_segmentation.data_utils.keypoint_data_loader import verify_keypoint_dataset

# Check if your dataset is properly formatted
is_valid = verify_keypoint_dataset("images/", "heatmaps/", n_keypoints=17)
```

## Advanced Usage

### Custom Loss Functions

```python
import keras.backend as K
from keras.losses import mean_squared_error

def custom_keypoint_loss(y_true, y_pred):
    # Example: Higher weight for keypoints, lower for background
    weight = 1.0 + 9.0 * K.cast(y_true > 0.1, 'float32')
    return K.mean(weight * K.square(y_true - y_pred))

# Use in training
model.train_keypoints(
    # ... other parameters ...
    loss_function=custom_keypoint_loss  # Pass function directly
)
```

### Multi-Scale Training

```python
# Train at multiple resolutions
resolutions = [(224, 224), (448, 448), (672, 672)]

for height, width in resolutions:
    model = keypoint_unet(n_keypoints=17, input_height=height, input_width=width)
    model.train_keypoints(
        train_images="images/",
        train_annotations="heatmaps/",
        input_height=height,
        input_width=width,
        n_keypoints=17,
        epochs=20
    )
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use smaller model variants
2. **Poor keypoint detection**: Try `weighted_mse` loss or increase heatmap sigma
3. **Inconsistent predictions**: Ensure proper data normalization [0-1] range

### Performance Tips

1. **Use appropriate sigma**: Heatmap spread should match keypoint precision needs
2. **Balance classes**: If some keypoints are rare, use weighted loss
3. **Data augmentation**: Apply rotation, scaling, and flipping to increase robustness
4. **Multi-stage training**: Train at low resolution first, then fine-tune at high resolution

## Complete Example

See `example_keypoint_regression.py` for a complete working example that:
- Generates synthetic keypoint data
- Trains a keypoint regression model
- Makes predictions and visualizes results
- Extracts keypoint coordinates

## Integration with Existing Code

The keypoint regression functionality is fully compatible with the existing keras-segmentation API. You can use the same training scripts, data loading utilities, and model architectures with minor modifications for keypoint-specific functionality.


