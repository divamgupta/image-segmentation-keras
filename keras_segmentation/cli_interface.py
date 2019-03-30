

import sys
import argparse
from . import train 
from . import predict 

from . import data_utils

from .data_utils.visualize_dataset import visualize_segmentation_dataset


def cli_train():

	parser = argparse.ArgumentParser()
	parser.add_argument("command", type = str   )
	parser.add_argument("--model_name", type = str  )
	parser.add_argument("--train_images", type = str   )
	parser.add_argument("--train_annotations", type = str   )

	parser.add_argument("--n_classes", type=int  )
	parser.add_argument("--input_height", type=int , default = None  )
	parser.add_argument("--input_width", type=int , default = None )

	parser.add_argument('--not_verify_dataset',action='store_false')
	parser.add_argument("--checkpoints_path", type = str  , default = None   )
	parser.add_argument("--epochs", type = int, default = 5 )
	parser.add_argument("--batch_size", type = int, default = 2 )
	

	parser.add_argument('--validate',action='store_true')
	parser.add_argument("--val_images", type = str , default = "")
	parser.add_argument("--val_annotations", type = str , default = "")
		
	parser.add_argument("--val_batch_size", type = int, default = 2 )
	parser.add_argument("--load_weights", type = str , default = None )
	parser.add_argument('--auto_resume_checkpoint',action='store_true')

	parser.add_argument("--steps_per_epoch", type = int, default = 512 )
	parser.add_argument("--optimizer_name", type = str , default = "adadelta")

	args = parser.parse_args()


	assert not args.model_name is None , "Please provide model_name"
	assert not args.train_images is None , "Please provide train_images"
	assert not args.train_annotations is None , "Please provide train_annotations"
	assert not args.n_classes is None , "Please provide n_classes"

	

	train.train( model=args.model_name   , 
		train_images=args.train_images  , 
		train_annotations=args.train_annotations  , 
		input_height=args.input_height  , 
		input_width=args.input_width  , 
		n_classes=args.n_classes ,
		verify_dataset=args.not_verify_dataset ,
		checkpoints_path=args.checkpoints_path  , 
		epochs = args.epochs ,
		batch_size = args.batch_size ,
		validate=args.validate  , 
		val_images=args.val_images  , 
		val_annotations=args.val_annotations  ,
		val_batch_size=args.val_batch_size  , 
		auto_resume_checkpoint=args.auto_resume_checkpoint  ,
		load_weights=args.load_weights  ,
		steps_per_epoch=args.steps_per_epoch ,
		optimizer_name=args.optimizer_name  
	)

def cli_predict():


	parser = argparse.ArgumentParser()
	parser.add_argument("command", type = str   )
	parser.add_argument("--checkpoints_path", type = str  )
	parser.add_argument("--input_path", type = str , default = "")
	parser.add_argument("--output_path", type = str , default = "")

	args = parser.parse_args()


	assert not args.checkpoints_path is None
	assert not args.input_path is None
	assert not args.output_path is None


	if ".jpg" in args.input_path or ".png" in args.input_path or ".jpeg" in args.input_path:
		predict.predict( inp=args.input_path  , out_fname=args.output_path  , checkpoints_path=args.checkpoints_path  )
	else:
		predict.predict_multiple( inp_dir=args.input_path , out_dir=args.output_path  , checkpoints_path=args.checkpoints_path   )


def cli_verify_dataset():

	parser = argparse.ArgumentParser()
	parser.add_argument("command", type = str   )
	parser.add_argument("--images_path", type = str   )
	parser.add_argument("--segs_path", type = str   )
	parser.add_argument("--n_classes", type=int  )

	args = parser.parse_args()

	data_utils.data_loader.verify_segmentation_dataset( args.images_path , args.segs_path , args.n_classes )

def cli_visualize_dataset():

	parser = argparse.ArgumentParser()
	parser.add_argument("command", type = str   )
	parser.add_argument("--images_path", type = str   )
	parser.add_argument("--segs_path", type = str   )
	parser.add_argument("--n_classes", type=int  )
	parser.add_argument('--do_augment',action='store_true')

	args = parser.parse_args()

	visualize_segmentation_dataset( args.images_path , args.segs_path ,  args.n_classes , do_augment=args.do_augment )







def main():
	assert len(sys.argv) >= 2 , "python -m keras_segmentation <command> <arguments>"

	command = sys.argv[1]

	if command == "train":
		cli_train()
	elif command == "predict":
		cli_predict()
	elif command == "verify_dataset":
		cli_verify_dataset()
	elif command == "visualize_dataset":
		cli_visualize_dataset()
	else:
		print("Invalid command " , command )

	print( command )

