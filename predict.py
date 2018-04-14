import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--epoch_number", type = int, default = 5 )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()


modelFns = {'vgg_segnet':Models.VGGSegnet.VGGSegnet,
			'vgg_unet':Models.VGGUnet.VGGUnet,
			'vgg_unet2':Models.VGGUnet.VGGUnet2,
			'fcn8':Models.FCN8.FCN8,
			'fcn32':Models.FCN32.FCN32}

modelFN = modelFns[args.model_name]

m = modelFN(args.n_classes, input_height=args.input_height, input_width=args.input_width)
m.load_weights(args.save_weights_path + "." + str(args.epoch_number))
m.compile(loss='categorical_crossentropy',
	optimizer= 'adadelta' ,
	metrics=['accuracy'])

images = glob.glob(os.path.join(args.test_images,"*.jpg")) + glob.glob(os.path.join(args.test_images,"*.png")) +  glob.glob(os.path.join(args.test_images,"*.jpeg"))
images.sort()

colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(args.n_classes)]

for imgName in images:
	outName = imgName.replace(args.test_images, args.output_path)
	X = LoadBatches.getImageArr(imgName, args.input_width, args.input_height)
	pr = m.predict(np.array([X]))[0]
	pr = pr.reshape((m.outputHeight, m.outputWidth, args.n_classes)).argmax(axis=2)
	seg_img = np.zeros((m.outputHeight , m.outputWidth , 3))
	for c in range(args.n_classes):
		seg_img[:,:,0] += ((pr[:,:] == c )*(colors[c][0])).astype('uint8')
		seg_img[:,:,1] += ((pr[:,:] == c )*(colors[c][1])).astype('uint8')
		seg_img[:,:,2] += ((pr[:,:] == c )*(colors[c][2])).astype('uint8')
	seg_img = cv2.resize(seg_img, (args.input_width, args.input_height))
	cv2.imwrite(outName, seg_img)
