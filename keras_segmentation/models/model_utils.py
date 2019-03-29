from keras.models import *
from keras.layers import *

import keras.backend as K
from types import MethodType


from .config import IMAGE_ORDERING
from ..train import train
from ..predict import predict , predict_multiple , evaluate


def resize_image( inp ,  s , data_format ):

	try:
		
		return Lambda( lambda x: K.resize_images(x, 
			height_factor=s[0], 
			width_factor=s[1], 
			data_format=data_format , 
			interpolation='bilinear') )( inp )

	except Exception as e:

		# if keras is old , then rely on the tf function ... sorry theono/cntk users . 
		assert data_format == 'channels_last'
		assert IMAGE_ORDERING == 'channels_last'

		import tensorflow as tf

		return Lambda( 
			lambda x: tf.image.resize_images(
				x , ( K.int_shape(x)[1]*s[0] ,K.int_shape(x)[2]*s[1] ))  
			)( inp )


def get_segmentation_model( input , output ):

	img_input = input
	o = output

	o_shape = Model(img_input , o ).output_shape
	i_shape = Model(img_input , o ).input_shape

	if IMAGE_ORDERING == 'channels_first':
		output_height = o_shape[2]
		output_width = o_shape[3]
		input_height = i_shape[2]
		input_width = i_shape[3]
		n_classes = o_shape[1]
		o = (Reshape((  -1  , output_height*output_width   )))(o)
		o = (Permute((2, 1)))(o)
	elif IMAGE_ORDERING == 'channels_last':
		output_height = o_shape[1]
		output_width = o_shape[2]
		input_height = i_shape[1]
		input_width = i_shape[2]
		n_classes = o_shape[3]
		o = (Reshape((   output_height*output_width , -1    )))(o)

	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.output_width = output_width
	model.output_height = output_height
	model.n_classes = n_classes
	model.input_height = input_height
	model.input_width = input_width
	model.model_name = ""

	model.train = MethodType( train , model )
	model.predict_segmentation = MethodType( predict , model )
	model.predict_multiple = MethodType( predict_multiple , model )
	model.evaluate_segmentation = MethodType( evaluate , model )


	return model 




