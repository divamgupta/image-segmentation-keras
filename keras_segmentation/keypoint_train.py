import json
import os

from .data_utils.keypoint_data_loader import keypoint_generator, \
    verify_keypoint_dataset
import six
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import glob
import sys

def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    # This is legacy code, there should always be a "checkpoint" file in your directory

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    if len(all_checkpoint_files) == 0:
        all_checkpoint_files = glob.glob(checkpoints_path + "*.*")
    all_checkpoint_files = [ff.replace(".index", "") for ff in
                            all_checkpoint_files]  # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))

    return latest_epoch_checkpoint


class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))


def train_keypoints(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_keypoints=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          optimizer_name='adam',
          do_augment=False,
          augmentation_name="aug_all",
          callbacks=None,
          custom_augmentation=None,
          other_inputs_paths=None,
          preprocessing=None,
          read_image_type=1,  # cv2.IMREAD_COLOR = 1 (rgb),
                              # cv2.IMREAD_GRAYSCALE = 0,
                              # cv2.IMREAD_UNCHANGED = -1 (4 channels like RGBA)
          loss_function='mse'  # Options: 'mse', 'binary_crossentropy', 'weighted_mse'
         ):
    from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_keypoints is not None), "Please provide the n_keypoints"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_keypoints, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_keypoints)

    n_keypoints = model.n_keypoints
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:

        # Choose loss function based on parameter
        if loss_function == 'mse':
            loss_k = 'mean_squared_error'
        elif loss_function == 'binary_crossentropy':
            loss_k = 'binary_crossentropy'
        elif loss_function == 'weighted_mse':
            # Custom weighted MSE that gives higher weight to positive keypoints
            def weighted_mse(y_true, y_pred):
                # Weight positive keypoints more heavily
                weight = 1.0 + 9.0 * y_true  # Weights: 1.0 for background, 10.0 for keypoints
                return K.mean(weight * K.square(y_true - y_pred))
            loss_k = weighted_mse
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=['mae'])  # Mean absolute error as additional metric

    if checkpoints_path is not None:
        config_file = checkpoints_path + "_config.json"
        dir_name = os.path.dirname(config_file)

        if ( not os.path.exists(dir_name) )  and len( dir_name ) > 0 :
            os.makedirs(dir_name)

        with open(config_file, "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_keypoints": n_keypoints,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    initial_epoch = 0

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

            initial_epoch = int(latest_checkpoint.split('.')[-1])

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_keypoint_dataset(train_images,
                                           train_annotations,
                                           n_keypoints)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_keypoint_dataset(val_images,
                                               val_annotations,
                                               n_keypoints)
            assert verified

    train_gen = keypoint_generator(
        train_images, train_annotations,  batch_size,  n_keypoints,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name,
        custom_augmentation=custom_augmentation, other_inputs_paths=other_inputs_paths,
        preprocessing=preprocessing, read_image_type=read_image_type)

    if validate:
        val_gen = keypoint_generator(
            val_images, val_annotations,  val_batch_size,
            n_keypoints, input_height, input_width, output_height, output_width,
            other_inputs_paths=other_inputs_paths,
            preprocessing=preprocessing, read_image_type=read_image_type)

    if callbacks is None and (not checkpoints_path is None) :
        default_callback = ModelCheckpoint(
                filepath=checkpoints_path + ".{epoch:05d}",
                save_weights_only=True,
                verbose=True
            )

        if sys.version_info[0] < 3: # for pyhton 2
            default_callback = CheckpointsCallback(checkpoints_path)

        callbacks = [
            default_callback
        ]

    if callbacks is None:
        callbacks = []

    if not validate:
        model.fit(train_gen, steps_per_epoch=steps_per_epoch,
                  epochs=epochs, callbacks=callbacks, initial_epoch=initial_epoch)
    else:
        model.fit(train_gen,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_gen,
                  validation_steps=val_steps_per_epoch,
                  epochs=epochs, callbacks=callbacks,
                  use_multiprocessing=gen_use_multiprocessing, initial_epoch=initial_epoch)
