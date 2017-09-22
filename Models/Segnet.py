
# todo upgrade to keras 2.0


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
# from keras.regularizers import ActivityRegularizer
from keras import backend as K





def segnet(nClasses , optimizer=None , input_height=360, input_width=480 ):

	kernel = 3
	filter_size = 64
	pad = 1
	pool_size = 2

	model = models.Sequential()
	model.add(Layer(input_shape=(3, input_height , input_width )))

	# encoder
	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))


	# decoder
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(512, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(256, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(128, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())


	model.add(Convolution2D( nClasses , 1, 1, border_mode='valid',))

	model.outputHeight = model.output_shape[-2]
	model.outputWidth = model.output_shape[-1]


	model.add(Reshape(( nClasses ,  model.output_shape[-2]*model.output_shape[-1]   ), input_shape=( nClasses , model.output_shape[-2], model.output_shape[-1]  )))
	
	model.add(Permute((2, 1)))
	model.add(Activation('softmax'))

	if not optimizer is None:
		model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )
	
	return model

