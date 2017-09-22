import argparse
import Models , LoadBatches


"""

THEANO_FLAGS=device=gpu0,floatX=float32  python train.py \
 --save_weights_path=weights/ex1 \
 --train_images="data/clothes_seg/prepped/images_prepped_train/" \
 --train_annotations="data/clothes_seg/prepped/annotations_prepped_train/" \
 --val_images="data/clothes_seg/prepped/images_prepped_test/" \
 --val_annotations="data/clothes_seg/prepped/annotations_prepped_test/" \
 --n_classes=10 \
 --input_height=800 \
 --input_width=550 


"""

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

args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

m = Models.VGGSegnet.VGGSegnet( n_classes  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( input_height , input_width )  )

output_height = m.outputHeight
output_width = m.outputWidth

G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


if validate:
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 512  , nb_epoch=1 )
		m.save_weights( save_weights_path + "." + str( ep ) )
else:
	for ep in range( epochs ):
		m.fit_generator( G , 512  , validation_data=G2 , nb_val_samples=200 ,  nb_epoch=1 )
		m.save_weights( save_weights_path + "." + str( ep )  )


