
import glob
import numpy as np
import cv2
import random
import argparse

from .augmentation import augment_seg
from .data_loader import get_pairs_from_paths

random.seed(0)
class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]



def visualize_segmentation_dataset( images_path , segs_path ,  n_classes , do_augment=False ):

	img_seg_pairs = get_pairs_from_paths( images_path , segs_path )

	colors = class_colors

	print("Press any key to navigate. ")
	for im_fn , seg_fn in img_seg_pairs :

		img = cv2.imread( im_fn )
		seg = cv2.imread( seg_fn )
		print("Found the following classes" , np.unique( seg ))

		seg_img = np.zeros_like( seg )

		if do_augment:
			img , seg[:,:,0] = augment_seg( img , seg[:,:,0] )

		for c in range(n_classes):
			seg_img[:,:,0] += ( (seg[:,:,0] == c )*( colors[c][0] )).astype('uint8')
			seg_img[:,:,1] += ((seg[:,:,0] == c )*( colors[c][1] )).astype('uint8')
			seg_img[:,:,2] += ((seg[:,:,0] == c )*( colors[c][2] )).astype('uint8')

		cv2.imshow("img" , img )
		cv2.imshow("seg_img" , seg_img )
		cv2.waitKey()



def visualize_segmentation_dataset_one( images_path , segs_path ,  n_classes , do_augment=False , no_show=False ):

	img_seg_pairs = get_pairs_from_paths( images_path , segs_path )

	colors = class_colors

	im_fn , seg_fn = random.choice(img_seg_pairs) 

	img = cv2.imread( im_fn )
	seg = cv2.imread( seg_fn )
	print("Found the following classes" , np.unique( seg ))

	seg_img = np.zeros_like( seg )

	if do_augment:
		img , seg[:,:,0] = augment_seg( img , seg[:,:,0] )

	for c in range(n_classes):
		seg_img[:,:,0] += ( (seg[:,:,0] == c )*( colors[c][0] )).astype('uint8')
		seg_img[:,:,1] += ((seg[:,:,0] == c )*( colors[c][1] )).astype('uint8')
		seg_img[:,:,2] += ((seg[:,:,0] == c )*( colors[c][2] )).astype('uint8')

	if not no_show:
		cv2.imshow("img" , img )
		cv2.imshow("seg_img" , seg_img )
		cv2.waitKey()

	return img , seg_img


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--images", type = str  )
	parser.add_argument("--annotations", type = str  )
	parser.add_argument("--n_classes", type=int )
	args = parser.parse_args()

	visualize_segmentation_dataset(args.images ,  args.annotations  ,  args.n_classes   ) 
