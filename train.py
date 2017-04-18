



import Models , LoadBatches


m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( input_height , input_width )  )

output_height = m.outputHeight
output_width = m.outputWidth


G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


m.fit_generator( G , 512  , nb_epoch=10 )


