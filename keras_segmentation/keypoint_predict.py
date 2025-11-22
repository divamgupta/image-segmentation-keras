import cv2
import numpy as np
import six
from .data_utils.keypoint_data_loader import get_image_array, get_keypoint_array
from .models.config import IMAGE_ORDERING


def predict_keypoints(model=None, inp=None, out_fname=None, keypoints_fname=None, overlay_img=False, show_legends=False, class_names=None, prediction_width=None, prediction_height=None, read_image_type=1):

    if model is None:
        raise ValueError("Model cannot be None")

    if inp is None:
        raise ValueError("Input image cannot be None")

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, read_image_type)

    n_classes = model.n_keypoints

    x = get_image_array(inp, model.input_width, model.input_height, ordering=IMAGE_ORDERING)

    pr = model.predict(np.array([x]))[0]

    # Reshape back to image dimensions
    pr = pr.reshape((model.output_height, model.output_width, n_classes))

    # Convert to uint8 for saving
    pr_uint8 = (pr * 255).astype(np.uint8)

    if out_fname is not None:
        # Save each keypoint heatmap as separate image
        for i in range(n_classes):
            keypoint_fname = f"{out_fname}_keypoint_{i}.png"
            cv2.imwrite(keypoint_fname, pr_uint8[:, :, i])

    if keypoints_fname is not None:
        # Save as numpy array
        np.save(keypoints_fname, pr)

    return pr


def predict_keypoint_coordinates(heatmap, threshold=0.5, max_peaks=1):
    """
    Extract keypoint coordinates from heatmap using weighted average or peak detection

    Args:
        heatmap: Single keypoint heatmap (H, W) with values in [0, 1]
        threshold: Minimum confidence threshold
        max_peaks: Maximum number of peaks to detect (1 for single keypoint)

    Returns:
        List of (x, y, confidence) tuples
    """
    if np.max(heatmap) < threshold:
        return []  # No keypoints above threshold

    # Find peaks in the heatmap
    if max_peaks == 1:
        # Use weighted average for single keypoint
        h, w = heatmap.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Weight by heatmap values
        total_weight = np.sum(heatmap)
        if total_weight > 0:
            x_weighted = np.sum(x_coords * heatmap) / total_weight
            y_weighted = np.sum(y_coords * heatmap) / total_weight
            confidence = np.max(heatmap)
            return [(x_weighted, y_weighted, confidence)]
        else:
            return []
    else:
        # Use peak detection for multiple keypoints (more complex, not implemented yet)
        # For now, return the max_peaks highest peaks
        flat_indices = np.argsort(heatmap.ravel())[-max_peaks:]
        peaks = []
        for idx in flat_indices:
            y, x = np.unravel_index(idx, heatmap.shape)
            confidence = heatmap[y, x]
            if confidence >= threshold:
                peaks.append((float(x), float(y), float(confidence)))
        return sorted(peaks, key=lambda x: x[2], reverse=True)


def predict_multiple_keypoints(model=None, inps=None, keypoints_fname=None):
    """
    Predict keypoints for multiple images
    """
    if model is None:
        raise ValueError("Model cannot be None")

    if inps is None or len(inps) == 0:
        raise ValueError("Input images cannot be None or empty")

    n_classes = model.n_keypoints

    # Process all images
    Xs = []
    for inp in inps:
        if isinstance(inp, six.string_types):
            inp = cv2.imread(inp, 1)
        x = get_image_array(inp, model.input_width, model.input_height, ordering=IMAGE_ORDERING)
        Xs.append(x)

    prs = model.predict(np.array(Xs))

    # Reshape all predictions
    predictions = []
    for pr in prs:
        pr_reshaped = pr.reshape((model.output_height, model.output_width, n_classes))
        predictions.append(pr_reshaped)

    if keypoints_fname is not None:
        np.save(keypoints_fname, np.array(predictions))

    return predictions
