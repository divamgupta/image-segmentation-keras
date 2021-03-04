from keras.models import *
from keras.layers import *

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder


# crop o1 wrt o2
def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height2 = o_shape2[2]
        output_width2 = o_shape2[3]
    else:
        output_height2 = o_shape2[1]
        output_width2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    if IMAGE_ORDERING == 'channels_first':
        output_height1 = o_shape1[2]
        output_width1 = o_shape1[3]
    else:
        output_height1 = o_shape1[1]
        output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format=IMAGE_ORDERING)(o2)

    if output_height1 > output_height2:
        o1 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format=IMAGE_ORDERING)(o2)

    return o1, o2


def fcn_8(n_classes, encoder=vanilla_encoder, input_height=416,
          input_width=608, channels=3):

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)

    o2 = f4
    o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                 data_format=IMAGE_ORDERING))(o2)

    o, o2 = crop(o, o2, img_input)

    o = Add()([o, o2])

    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o2 = f3
    o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                 data_format=IMAGE_ORDERING))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add( name="seg_feats" )([o2, o])

    o = Conv2DTranspose(n_classes, kernel_size=(16, 16),  strides=(
        8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)
    model.model_name = "fcn_8"
    return model


def fcn_32(n_classes, encoder=vanilla_encoder, input_height=416,
           input_width=608, channels=3):

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                data_format=IMAGE_ORDERING , name="seg_feats" ))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(64, 64),  strides=(
        32, 32), use_bias=False,  data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)
    model.model_name = "fcn_32"
    return model


def fcn_8_vgg(n_classes,  input_height=416, input_width=608, channels=3):
    model = fcn_8(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_8_vgg"
    return model


def fcn_32_vgg(n_classes,  input_height=416, input_width=608, channels=3):
    model = fcn_32(n_classes, get_vgg_encoder,
                   input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_32_vgg"
    return model


def fcn_8_resnet50(n_classes,  input_height=416, input_width=608, channels=3):
    model = fcn_8(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_8_resnet50"
    return model


def fcn_32_resnet50(n_classes,  input_height=416, input_width=608, channels=3):
    model = fcn_32(n_classes, get_resnet50_encoder,
                   input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_32_resnet50"
    return model


def fcn_8_mobilenet(n_classes,  input_height=224, input_width=224, channels=3):
    model = fcn_8(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_8_mobilenet"
    return model


def fcn_32_mobilenet(n_classes,  input_height=224, input_width=224, channels=3):
    model = fcn_32(n_classes, get_mobilenet_encoder,
                   input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_32_mobilenet"
    return model


if __name__ == '__main__':
    m = fcn_8(101)
    m = fcn_32(101)
