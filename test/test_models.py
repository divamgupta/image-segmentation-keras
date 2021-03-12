import numpy as np
import tempfile

import sys 

import keras

from keras_segmentation.models import all_models
from keras_segmentation.data_utils.data_loader import \
    verify_segmentation_dataset, image_segmentation_generator
from keras_segmentation.predict import predict_multiple, predict, evaluate


from keras_segmentation.model_compression import perform_distilation
from keras_segmentation.pretrained import pspnet_50_ADE_20K  

tr_im = "test/example_dataset/images_prepped_train"
tr_an = "test/example_dataset/annotations_prepped_train"
te_im = "test/example_dataset/images_prepped_test"
te_an = "test/example_dataset/annotations_prepped_test"



def test_models():

    n_c = 100

    models = [ ( "unet_mini" , 124 , 156   )  , ( "vgg_unet" , 224 , 224*2   ) , 
        ( 'resnet50_pspnet', 192*2 , 192*3 ) , ( 'mobilenet_unet', 224 , 224 ), ( 'mobilenet_unet', 224+32 , 224+32 ),( 'segnet', 224 , 224*2 ),( 'vgg_segnet', 224 , 224*2 ) ,( 'fcn_32', 224 , 224*2 ) ,( 'fcn_8_vgg', 224 , 224*2 )   ]

    for model_name  , h , w in models:
        m = all_models.model_from_name[model_name]( n_c, input_height=h, input_width=w)

        m.train(train_images=tr_im,
            train_annotations=tr_an,
            steps_per_epoch=2,
            epochs=2 )

        keras.backend.clear_session()





def test_verify():
    verify_segmentation_dataset(tr_im, tr_an, 50)


def test_datag():
    g = image_segmentation_generator(images_path=tr_im, segs_path=tr_an,
                                     batch_size=3,  n_classes=50,
                                     input_height=224, input_width=324,
                                     output_height=114, output_width=134,
                                     do_augment=False)

    x, y = next(g)
    assert x.shape[0] == 3
    assert y.shape[0] == 3
    assert y.shape[-1] == 50


# with augmentation
def test_datag2():
    g = image_segmentation_generator(images_path=tr_im, segs_path=tr_an,
                                     batch_size=3,  n_classes=50,
                                     input_height=224, input_width=324,
                                     output_height=114, output_width=134,
                                     do_augment=True)

    x, y = next(g)
    assert x.shape[0] == 3
    assert y.shape[0] == 3
    assert y.shape[-1] == 50


def test_model():
    model_name = "fcn_8"
    h = 224
    w = 256
    n_c = 100
    check_path = tempfile.mktemp()

    m = all_models.model_from_name[model_name](
                                            n_c, input_height=h, input_width=w)

    m.train(train_images=tr_im,
            train_annotations=tr_an,
            steps_per_epoch=2,
            epochs=2,
            checkpoints_path=check_path
            )

    m.train(train_images=tr_im,
            train_annotations=tr_an,
            steps_per_epoch=2,
            epochs=2,
            checkpoints_path=check_path,
            augmentation_name='aug_geometric', do_augment=True
            )

    m.predict_segmentation(np.zeros((h, w, 3))).shape

    predict_multiple(
        inp_dir=te_im, checkpoints_path=check_path, out_dir="/tmp")
    predict_multiple(inps=[np.zeros((h, w, 3))]*3,
                     checkpoints_path=check_path, out_dir="/tmp")

    ev = m.evaluate_segmentation(inp_images_dir=te_im, annotations_dir=te_an)
    assert ev['frequency_weighted_IU'] > 0.01
    print(ev)
    o = predict(inp=np.zeros((h, w, 3)), checkpoints_path=check_path)

    o = predict(inp=np.zeros((h, w, 3)), checkpoints_path=check_path,
                overlay_img=True, class_names=['nn']*n_c, show_legends=True)
    print("pr")

    o.shape

    ev = evaluate(inp_images_dir=te_im,
                  annotations_dir=te_an,
                  checkpoints_path=check_path)
    assert ev['frequency_weighted_IU'] > 0.01


def test_kd():

    if sys.version_info.major < 3:
        # KD wont work with python 2 
        return 

    model_name = "fcn_8"
    h = 224
    w = 256
    n_c = 100
    check_path1 = tempfile.mktemp()

    m1 = all_models.model_from_name[model_name](
                                            n_c, input_height=h, input_width=w)



    model_name = "unet_mini"
    h = 124
    w = 156
    n_c = 100
    check_path2 = tempfile.mktemp()

    m2 = all_models.model_from_name[model_name](
                                            n_c, input_height=h, input_width=w)


    m1.train(train_images=tr_im,
            train_annotations=tr_an,
            steps_per_epoch=2,
            epochs=2,
            checkpoints_path=check_path1
            )

    perform_distilation(m1 ,m2, tr_im , distilation_loss='kl' , 
                        batch_size =2 ,checkpoints_path=check_path2  , epochs = 2 , steps_per_epoch=2, )


    perform_distilation(m1 ,m2, tr_im , distilation_loss='l2' , 
                        batch_size =2 ,checkpoints_path=check_path2  , epochs = 2 , steps_per_epoch=2, )


    perform_distilation(m1 ,m2, tr_im , distilation_loss='l2' , 
                        batch_size =2 ,checkpoints_path=check_path2  , epochs = 2 , steps_per_epoch=2, feats_distilation_loss='pa' )










def test_pretrained():
    

    model = pspnet_50_ADE_20K() 

    out = model.predict_segmentation(
        inp=te_im+"/0016E5_07959.png",
        out_fname="/tmp/out.png"
    )








# def test_models():


# 	unet_models = [  , models.unet.vgg_unet , models.unet.resnet50_unet ]
# 	args = [(101, 416, 608), (101, 224 ,224), (101, 256, 256), (2, 32*4, 32*5)]
# 	en_level = [ 1,2,3,4 ]

# 	for mf in unet_models:
# 		for en in en_level:
# 			for ar in args:
# 				m = mf( *ar , encoder_level=en )


# 	m = models.unet.mobilenet_unet( 55 )
# 	for ar in args:
# 		m = unet_mini( *ar )
