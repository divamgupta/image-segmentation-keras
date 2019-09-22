from types import MethodType

from keras.models import *
from keras.layers import *
import keras.backend as K
from tqdm import tqdm

from .config import IMAGE_ORDERING
from ..train import train
from ..predict import predict, predict_multiple, evaluate


# source m1 , dest m2
def transfer_weights(m1, m2, verbose=True):

    assert len(m1.layers) == len(
        m2.layers), "Both models should have same number of layers"

    nSet = 0
    nNotSet = 0

    if verbose:
        print("Copying weights ")
        bar = tqdm(zip(m1.layers, m2.layers))
    else:
        bar = zip(m1.layers, m2.layers)

    for l, ll in bar:

        if not any([w.shape != ww.shape for w, ww in zip(list(l.weights),
                                                         list(ll.weights))]):
            if len(list(l.weights)) > 0:
                ll.set_weights(l.get_weights())
                nSet += 1
        else:
            nNotSet += 1

    if verbose:
        print("Copied weights of %d layers and skipped %d layers" %
              (nSet, nNotSet))


def resize_image(inp,  s, data_format):

    try:

        return Lambda(lambda x: K.resize_images(x,
                                                height_factor=s[0],
                                                width_factor=s[1],
                                                data_format=data_format,
                                                interpolation='bilinear'))(inp)

    except Exception as e:
        # if keras is old, then rely on the tf function
        # Sorry theano/cntk users!!!
        assert data_format == 'channels_last'
        assert IMAGE_ORDERING == 'channels_last'

        import tensorflow as tf

        return Lambda(
            lambda x: tf.image.resize_images(
                x, (K.int_shape(x)[1]*s[0], K.int_shape(x)[2]*s[1]))
        )(inp)


def get_segmentation_model(input, output):

    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        o = (Reshape((-1, output_height*output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
        o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    model.predict_multiple = MethodType(predict_multiple, model)
    model.evaluate_segmentation = MethodType(evaluate, model)

    return model
