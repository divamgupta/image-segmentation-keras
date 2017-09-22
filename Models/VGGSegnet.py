





from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Merge
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam , SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras import backend as K

import os

import Utils


# for input(360,480) output will be  ( 170 , 240)

# input_image_size -> ( height , width )
def VGGSegnet( n_classes  , use_vgg_weights=True ,  optimizer=None , input_image_size=(224 , 224 )  ):

	model = Sequential()
	
	model.add(ZeroPadding2D((1,1),input_shape=(  3 , input_image_size[0]  , input_image_size[1] )))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	if use_vgg_weights:
		weights_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/vgg16_weights.h5"
		Utils.loadWeightsPartial( model ,  weights_path , len(model.layers ) )
	
	model.add( ZeroPadding2D(padding=(1,1)))
	model.add( Convolution2D(512, 3, 3, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(2,2)))
	model.add( ZeroPadding2D(padding=(1,1)))
	model.add( Convolution2D(256, 3, 3, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(2,2)))
	model.add( ZeroPadding2D(padding=(1,1)))
	model.add( Convolution2D(128, 3, 3, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(2,2)))
	model.add( ZeroPadding2D(padding=(1,1)))
	model.add( Convolution2D(64, 3, 3, border_mode='valid'))
	model.add( BatchNormalization())

	model.add(Convolution2D( n_classes , 1, 1, border_mode='valid',))

	model.outputHeight = model.output_shape[-2]
	model.outputWidth = model.output_shape[-1]
	# print "Model Output shape - " , model.output_shape
	model.add(Reshape(( n_classes ,  model.output_shape[-2]*model.output_shape[-1]   ), input_shape=( n_classes , model.output_shape[-2], model.output_shape[-1]  )))
	model.add(Permute((2, 1)))

	model.add(Activation('softmax'))

	if not optimizer is None:
		model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )

	return model




