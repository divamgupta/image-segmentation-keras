import pytest
from keras_segmentation import  models 
from  keras_segmentation.models.config import IMAGE_ORDERING


def test_models():


	unet_models = [  , models.unet.vgg_unet , models.unet.resnet50_unet ]
	args = [ ( 101, 416 , 608)  , ( 101, 224 , 224)  , ( 101, 256  , 256 ) , ( 2, 32*4  , 32*5 )  ]
	en_level = [ 1,2,3,4 ]

	for mf in unet_models:
		for en in en_level:
			for ar in args:
				m = mf( *ar , encoder_level=en )

	


	m = models.unet.mobilenet_unet( 55 )
	for ar in args:
		m = unet_mini( *ar )






