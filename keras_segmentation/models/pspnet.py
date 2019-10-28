import numpy as np
import keras
from keras.models import *
from keras.layers import *
import keras.backend as K

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model, resize_image
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder


if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def pool_block(feats, pool_factor):

    if IMAGE_ORDERING == 'channels_first':
        h = K.int_shape(feats)[2]
        w = K.int_shape(feats)[3]
    elif IMAGE_ORDERING == 'channels_last':
        h = K.int_shape(feats)[1]
        w = K.int_shape(feats)[2]

    pool_size = strides = [
        int(np.round(float(h) / pool_factor)),
        int(np.round(float(w) / pool_factor))]

    x = AveragePooling2D(pool_size, data_format=IMAGE_ORDERING,
                         strides=strides, padding='same')(feats)
    x = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING,
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = resize_image(x, strides, data_format=IMAGE_ORDERING)

    return x


def _pspnet(n_classes, encoder,  input_height=384, input_width=576):

    assert input_height % 192 == 0
    assert input_width % 192 == 0

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate(axis=MERGE_AXIS)(pool_outs)

    o = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(n_classes, (3, 3), data_format=IMAGE_ORDERING,
               padding='same')(o)
    o = resize_image(o, (8, 8), data_format=IMAGE_ORDERING)

    model = get_segmentation_model(img_input, o)
    return model


def pspnet(n_classes,  input_height=384, input_width=576):

    model = _pspnet(n_classes, vanilla_encoder,
                    input_height=input_height, input_width=input_width)
    model.model_name = "pspnet"
    return model


def vgg_pspnet(n_classes,  input_height=384, input_width=576):

    model = _pspnet(n_classes, get_vgg_encoder,
                    input_height=input_height, input_width=input_width)
    model.model_name = "vgg_pspnet"
    return model


def resnet50_pspnet(n_classes,  input_height=384, input_width=576):

    model = _pspnet(n_classes, get_resnet50_encoder,
                    input_height=input_height, input_width=input_width)
    model.model_name = "resnet50_pspnet"
    return model


def pspnet_50(n_classes,  input_height=473, input_width=473):
    from ._pspnet_2 import _build_pspnet

    nb_classes = n_classes
    resnet_layers = 50
    input_shape = (input_height, input_width)
    model = _build_pspnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape)
    model.model_name = "pspnet_50"
    return model


def pspnet_101(n_classes,  input_height=473, input_width=473):
    from ._pspnet_2 import _build_pspnet

    nb_classes = n_classes
    resnet_layers = 101
    input_shape = (input_height, input_width)
    model = _build_pspnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape)
    model.model_name = "pspnet_101"
    return model


# def mobilenet_pspnet( n_classes ,  input_height=224, input_width=224 ):

# 	model =  _pspnet(n_classes, get_mobilenet_encoder,
#                    input_height=input_height, input_width=input_width)
# 	model.model_name = "mobilenet_pspnet"
# 	return model


if __name__ == '__main__':

    m = _pspnet(101, vanilla_encoder)
    # m = _pspnet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
    m = _pspnet(101, get_vgg_encoder)
    m = _pspnet(101, get_resnet50_encoder)
