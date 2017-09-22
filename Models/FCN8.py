
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
# assert 0 == 1 # fc weights into the 1x1 convs  , get_upsampling_weight 



from keras.models import *
from keras.layers import *


import os
file_path = os.path.dirname( os.path.abspath(__file__) )

VGG_Weights_path = file_path+"/../../data/vgg16_weights_th_dim_ordering_th_kernels.h5"


# for input(360,480) output will be  ( 170 , 240)

# input_image_size -> ( height , width )

# crop o1 wrt o2
def crop( o1 , o2 , i  ):
	o_shape2 = Model( i  , o2 ).output_shape
	outputHeight2 = o_shape2[2]
	outputWidth2 = o_shape2[3]

	o_shape1 = Model( i  , o1 ).output_shape
	outputHeight1 = o_shape1[2]
	outputWidth1 = o_shape1[3]

	cx = abs( outputWidth1 - outputWidth2 )
	cy = abs( outputHeight2 - outputHeight1 )

	if outputWidth1 > outputWidth2:
		o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format='channels_first'  )(o1)
	else:
		o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format='channels_first'  )(o2)
	
	if outputHeight1 > outputHeight2 :
		o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format='channels_first'  )(o1)
	else:
		o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format='channels_first'  )(o2)

	return o1 , o2 

def FCN8( nClasses ,  input_height=416, input_width=608 , vgg_level=3):

	# assert input_height%32 == 0
	# assert input_width%32 == 0

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	n_classes = 3
	img_input = Input(shape=(3,input_height,input_width))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first' )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first' )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_first' )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_first' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first' )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_first' )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_first' )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_first' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_first' )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_first' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_first' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_first' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_first' )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_first' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_first' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_first' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_first' )(x)
	f5 = x

	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense( 1024 , activation='softmax', name='predictions')(x)

	vgg  = Model(  img_input , x  )
	# vgg.load_weights(VGG_Weights_path)

	o = f5

	o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format='channels_first'))(o)
	o = Dropout(0.5)(o)
	o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format='channels_first'))(o)
	o = Dropout(0.5)(o)

	o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format='channels_first'))(o)
	o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format='channels_first' )(o)

	o2 = f4
	o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format='channels_first'))(o2)
	
	o , o2 = crop( o , o2 , img_input )
	
	o = Add()([ o , o2 ])

	o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format='channels_first' )(o)
	o2 = f3 
	o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format='channels_first'))(o2)
	o2 , o = crop( o2 , o , img_input )
	o  = Add()([ o2 , o ])


	o = Conv2DTranspose( nClasses , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False, data_format='channels_first' )(o)
	
	o_shape = Model(img_input , o ).output_shape
	
	outputHeight = o_shape[2]
	outputWidth = o_shape[3]

	o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	return model



if __name__ == '__main__':
	m = FCN8( 101 )
	from keras.utils import plot_model
	plot_model( m , show_shapes=True , to_file='model.png')
