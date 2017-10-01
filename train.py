import argparse
import Models , LoadBatches


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()


if args.validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = {'vgg_segnet':Models.VGGSegnet.VGGSegnet,
			'vgg_unet':Models.VGGUnet.VGGUnet,
			'vgg_unet2':Models.VGGUnet.VGGUnet2,
			'fcn8':Models.FCN8.FCN8,
			'fcn32':Models.FCN32.fcn32}

modelFN = modelFns[args.model_name]

m = modelFN( args.n_classes , input_height=args.input_height, input_width=args.input_width   )
m.compile(loss='categorical_crossentropy',
      optimizer= args.optimizer_name ,
      metrics=['accuracy'])


if len( args.load_weights ) > 0:
	m.load_weights(args.load_weights)


print "Model output shape",  m.output_shape

output_height = m.outputHeight
output_width = m.outputWidth

G = LoadBatches.imageSegmentationGenerator( args.train_images , args.train_annotations ,  args.batch_size,  args.n_classes , args.input_height , args.input_width , output_height , output_width   )


if args.validate:
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  args.n_classes , args.input_height , args.input_width , output_height , output_width   )

if not args.validate:
	for ep in range( args.epochs ):
		m.fit_generator( G , 512  , epochs=1 )
		m.save_weights( args.save_weights_path + "." + str( ep ) )
		m.save( args.save_weights_path + ".model." + str( ep ) )
else:
	for ep in range( args.epochs ):
		m.fit_generator( G , 512  , validation_data=G2 , validation_steps=200 ,  epochs=1 )
		m.save_weights( args.save_weights_path + "." + str( ep )  )
		m.save( args.save_weights_path + ".model." + str( ep ) )
