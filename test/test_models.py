import pytest
from keras_segmentation import  models 
from  keras_segmentation.models.config import IMAGE_ORDERING
import random
import numpy as np

from keras_segmentation.data_utils.data_loader import verify_segmentation_dataset , image_segmentation_generator

from keras_segmentation.predict import predict_multiple , predict

tr_im = "test/example_dataset/images_prepped_train"
tr_an = "test/example_dataset/annotations_prepped_train"
te_im = "test/example_dataset/images_prepped_test"
te_an = "test/example_dataset/annotations_prepped_test"


def test_verify():
	verify_segmentation_dataset( tr_im , tr_an , 50)


def test_datag():
	g = image_segmentation_generator( images_path=tr_im , segs_path=tr_an ,  batch_size=3 ,  n_classes=50 , input_height=224 , input_width=324 , output_height=114 , output_width=134  , do_augment=False )

	x , y = next(g)
	assert x.shape[0] == 3
	assert y.shape[0] == 3
	assert y.shape[-1] == 50



def test_model( ):
	model_name = "fcn_8"
	h = 224
	w = 256
	n_c = 100
	check_path = "/tmp/%d"%(random.randint(0,199999))


	m = models.model_from_name[ model_name ]( n_c ,  input_height=h, input_width=w )

	m.train(train_images=tr_im ,
		train_annotations=tr_an  , 
		steps_per_epoch=2 , 
		epochs=2,
		checkpoints_path=check_path 
	)

	m.predict_segmentation( np.zeros((h , w , 3 )) ).shape


	predict_multiple(inp_dir=te_im  , checkpoints_path=check_path  , out_dir="/tmp")
	predict_multiple(inps=[np.zeros((h , w , 3 ))]*3  , checkpoints_path=check_path  , out_dir="/tmp")

	o = predict(inp=np.zeros((h , w , 3 ))   , checkpoints_path=check_path   )
	o.shape






# def test_models():


# 	unet_models = [  , models.unet.vgg_unet , models.unet.resnet50_unet ]
# 	args = [ ( 101, 416 , 608)  , ( 101, 224 , 224)  , ( 101, 256  , 256 ) , ( 2, 32*4  , 32*5 )  ]
# 	en_level = [ 1,2,3,4 ]

# 	for mf in unet_models:
# 		for en in en_level:
# 			for ar in args:
# 				m = mf( *ar , encoder_level=en )

	


# 	m = models.unet.mobilenet_unet( 55 )
# 	for ar in args:
# 		m = unet_mini( *ar )






