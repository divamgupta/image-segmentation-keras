import keras
import tensorflow as tf
from .models.all_models import model_from_name


def model_from_checkpoint_path(model_config, latest_weights):

    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    model.load_weights(latest_weights)
    return model


def resnet_pspnet_VOC12_v0_1():

    model_config = {
        "output_height": 96,
        "input_height": 384,
        "input_width": 576,
        "n_classes": 151,
        "model_class": "resnet50_pspnet",
        "output_width": 144
    }

    REPO_URL = "https://github.com/divamgupta/image-segmentation-keras"
    MODEL_PATH = "pretrained_model_1/r2_voc12_resnetpspnet_384x576.24"
    model_url = "{0}/releases/download/{1}".format(REPO_URL, MODEL_PATH)
    latest_weights = tf.keras.utils.get_file(model_url.split("/")[-1], model_url)

    return model_from_checkpoint_path(model_config, latest_weights)


# pretrained model converted from caffe by Vladkryvoruchko ... thanks !
def pspnet_50_ADE_20K():

    model_config = {
        "input_height": 473,
        "input_width": 473,
        "n_classes": 150,
        "model_class": "pspnet_50",
    }

    model_url = "https://www.dropbox.com/s/" \
                "0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1"
    latest_weights = tf.keras.utils.get_file("pspnet50_ade20k.h5", model_url)

    return model_from_checkpoint_path(model_config, latest_weights)


def pspnet_101_cityscapes():

    model_config = {
        "input_height": 713,
        "input_width": 713,
        "n_classes": 19,
        "model_class": "pspnet_101",
    }

    model_url = "https://www.dropbox.com/s/" \
                "c17g94n946tpalb/pspnet101_cityscapes.h5?dl=1"
    latest_weights = tf.keras.utils.get_file("pspnet101_cityscapes.h5", model_url)

    return model_from_checkpoint_path(model_config, latest_weights)


def pspnet_101_voc12():

    model_config = {
        "input_height": 473,
        "input_width": 473,
        "n_classes": 21,
        "model_class": "pspnet_101",
    }

    model_url = "https://www.dropbox.com/s/" \
                "uvqj2cjo4b9c5wg/pspnet101_voc2012.h5?dl=1"
    latest_weights = tf.keras.utils.get_file("pspnet101_voc2012.h5", model_url)

    return model_from_checkpoint_path(model_config, latest_weights)
