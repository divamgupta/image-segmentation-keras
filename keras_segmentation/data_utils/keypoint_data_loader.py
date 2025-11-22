import itertools
import os
import random
import six
import numpy as np
import cv2

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")

    def tqdm(iter):
        return iter

from ..models.config import IMAGE_ORDERING
from .augmentation import augment_seg, custom_augment_seg

DATA_LOADER_SEED = 0

random.seed(DATA_LOADER_SEED)

ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
ACCEPTABLE_KEYPOINT_FORMATS = [".png", ".npy"]


def get_image_list_from_path(images_path ):
    image_files = []
    for dir_entry in os.listdir(images_path):
            if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                    os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
                file_name, file_extension = os.path.splitext(dir_entry)
                image_files.append(os.path.join(images_path, dir_entry))
    return image_files


def get_keypoint_pairs_from_paths(images_path, keypoints_path):
    """ Find all the images from the images_path directory and
        the keypoint heatmaps from the keypoints_path directory
        while checking integrity of data """

    image_files = []
    keypoint_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))

    for dir_entry in os.listdir(keypoints_path):
        if os.path.isfile(os.path.join(keypoints_path, dir_entry)):
            file_name, file_extension = os.path.splitext(dir_entry)
            if file_extension in ACCEPTABLE_KEYPOINT_FORMATS:
                full_dir_entry = os.path.join(keypoints_path, dir_entry)
                if file_name in keypoint_files:
                    raise ValueError("Keypoint file with filename {0}"
                                    " already exists and is ambiguous to"
                                    " resolve with path {1}."
                                    " Please remove or rename the latter."
                                    .format(file_name, full_dir_entry))

                keypoint_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    # Match the images and keypoints
    for image_file, _, image_full_path in image_files:
        if image_file in keypoint_files:
            keypoint_extension, keypoint_full_path = keypoint_files[image_file]
            return_value.append((image_full_path, keypoint_full_path))

    return return_value


def get_image_array(image_input, width, height,
                    imgNorm="sub_mean", ordering='channels_last'):

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise ValueError("get_image_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise ValueError("get_image_array: Can't process input type {0}".format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)

        means = [103.939, 116.779, 123.68]

        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def get_keypoint_array(keypoint_input, n_keypoints, width, height):
    """ Load keypoint heatmap array from input """

    if type(keypoint_input) is np.ndarray:
        # It is already an array, use it as it is
        heatmap = keypoint_input
    elif isinstance(keypoint_input, six.string_types):
        if not os.path.isfile(keypoint_input):
            raise ValueError("get_keypoint_array: path {0} doesn't exist".format(keypoint_input))

        if keypoint_input.endswith('.npy'):
            # Load numpy array directly
            heatmap = np.load(keypoint_input)
        else:
            # Load image file
            heatmap = cv2.imread(keypoint_input, cv2.IMREAD_UNCHANGED)

            # If it's a single channel image, assume it's for one keypoint
            if len(heatmap.shape) == 2:
                heatmap = heatmap[:, :, np.newaxis]
            elif len(heatmap.shape) == 3 and heatmap.shape[2] == 1:
                pass  # Already single channel
            elif len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
                # RGB image - assume each channel represents a different keypoint
                if n_keypoints == 3:
                    pass
                else:
                    # Take mean across channels or first channel
                    heatmap = np.mean(heatmap, axis=2, keepdims=True)
            else:
                raise ValueError(f"Unsupported keypoint image format with {heatmap.shape[2]} channels")

            # Normalize to [0, 1] range if needed
            if heatmap.dtype != np.float32:
                heatmap = heatmap.astype(np.float32)
                if np.max(heatmap) > 1.0:
                    heatmap = heatmap / 255.0
    else:
        raise ValueError("get_keypoint_array: Can't process input type {0}".format(str(type(keypoint_input))))

    # Resize to target dimensions
    if heatmap.shape[0] != height or heatmap.shape[1] != width:
        resized_channels = []
        for c in range(heatmap.shape[2]):
            resized = cv2.resize(heatmap[:, :, c], (width, height), interpolation=cv2.INTER_LINEAR)
            resized_channels.append(resized)
        heatmap = np.stack(resized_channels, axis=2)

    # Ensure we have the right number of keypoints
    if heatmap.shape[2] != n_keypoints:
        if heatmap.shape[2] == 1 and n_keypoints > 1:
            # Repeat the single channel for all keypoints
            heatmap = np.repeat(heatmap, n_keypoints, axis=2)
        elif heatmap.shape[2] > n_keypoints:
            # Take only the first n_keypoints channels
            heatmap = heatmap[:, :, :n_keypoints]
        else:
            raise ValueError(f"Keypoint array has {heatmap.shape[2]} channels but model expects {n_keypoints}")

    # Flatten to (height*width, n_keypoints)
    heatmap = np.reshape(heatmap, (height*width, n_keypoints))

    return heatmap


def verify_keypoint_dataset(images_path, keypoints_path, n_keypoints, show_all_errors=False):
    try:
        img_keypoint_pairs = get_keypoint_pairs_from_paths(images_path, keypoints_path)
        if not len(img_keypoint_pairs):
            print("Couldn't load any data from images_path: "
                  "{0} and keypoints path: {1}"
                  .format(images_path, keypoints_path))
            return False

        return_value = True
        for im_fn, kp_fn in tqdm(img_keypoint_pairs):
            img = cv2.imread(im_fn)
            keypoint = get_keypoint_array(kp_fn, n_keypoints, 224, 224)  # Use dummy dimensions for verification

            # Check that keypoint values are in valid range [0, 1]
            if np.min(keypoint) < 0.0 or np.max(keypoint) > 1.0:
                return_value = False
                print("The keypoint values in {0} are not in range [0, 1]. "
                      "Found min: {1}, max: {2}"
                      .format(kp_fn, np.min(keypoint), np.max(keypoint)))
                if not show_all_errors:
                    break

        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except Exception as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False


def keypoint_generator(images_path, keypoints_path, batch_size,
                       n_keypoints, input_height, input_width,
                       output_height, output_width,
                       do_augment=False,
                       augmentation_name="aug_all",
                       custom_augmentation=None,
                       other_inputs_paths=None, preprocessing=None,
                       read_image_type=cv2.IMREAD_COLOR):

    img_keypoint_pairs = get_keypoint_pairs_from_paths(images_path, keypoints_path, other_inputs_paths=other_inputs_paths)
    random.shuffle(img_keypoint_pairs)
    zipped = itertools.cycle(img_keypoint_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            if other_inputs_paths is None:

                im, kp = next(zipped)
                im = cv2.imread(im, read_image_type)
                kp_array = get_keypoint_array(kp, n_keypoints, output_width, output_height)

                if do_augment:
                    # For now, skip augmentation for keypoints - can be added later
                    pass

                if preprocessing is not None:
                    im = preprocessing(im)

                X.append(get_image_array(im, input_width,
                                         input_height, ordering=IMAGE_ORDERING))
                Y.append(kp_array)
            else:
                # Handle multiple inputs - similar to original data loader
                im, kp, others = next(zipped)

                im = cv2.imread(im, read_image_type)
                kp_array = get_keypoint_array(kp, n_keypoints, output_width, output_height)

                oth = []
                for f in others:
                    oth.append(cv2.imread(f, read_image_type))

                if do_augment:
                    # Skip augmentation for now
                    ims = [im]
                    ims.extend(oth)
                else:
                    ims = [im]
                    ims.extend(oth)

                oth = []
                for i, image in enumerate(ims):
                    oth_im = get_image_array(image, input_width,
                                             input_height, ordering=IMAGE_ORDERING)

                    if preprocessing is not None:
                        if isinstance(preprocessing, list):
                            oth_im = preprocessing[i](oth_im)
                        else:
                            oth_im = preprocessing(oth_im)

                    oth.append(oth_im)

                X.append(oth)
                Y.append(kp_array)

        yield np.array(X), np.array(Y)
