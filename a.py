# from keras_segmentation.data_utils.visualize_dataset import visualize_segmentation_dataset
# from keras_segmentation.data_utils.data_loader import verify_segmentation_dataset , image_segmentation_generator



a= "/Users/divamgupta/Downloads/dataset1/images_prepped_train"
b="/Users/divamgupta/Downloads/dataset1/annotations_prepped_train"

# # verify_segmentation_dataset( a , b , 50)

# # visualize_segmentation_dataset( a , b , 50 , do_augment=False  )


# g = image_segmentation_generator( images_path=a , segs_path=b ,  batch_size=3 ,  n_classes=50 , input_height=224 , input_width=324 , output_height=114 , output_width=134  , do_augment=False )


# from keras_segmentation import  models

# m = models.unet.unet(50)


immm = "/Users/divamgupta/Downloads/screencapture-mutualfund-adityabirlacapital-Portal-Smartlink-Acknowldgement-SIPPurchaseAcknowldgementDetails-2019-03-25-14_53_53.png"

# print m.predict_segmentation(immm).shape

# m.train(train_images=a ,
# 	train_annotations=b , 
# 	steps_per_epoch=3 , 
# 	checkpoints_path="/tmp/hoho"
# )


from keras_segmentation.predict import predict , predict_multiple
from keras_segmentation.train import train


# train( model='unet' , 
# 	train_images = a,
# 	train_annotations=b,
# 	checkpoints_path="/tmp/hoho",
# 	n_classes=50 , 
# 	steps_per_epoch=3 
# )

print predict_multiple(inps= [immm]*4 , checkpoints_path="/tmp/hoho" , out_dir="/tmp/bhobho")




