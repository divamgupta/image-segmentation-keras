
import numpy as np
import cv2
import glob
import itertools


def getImageArr( path , width , height , imgNorm="sub_mean" ):

	try:
		img = cv2.imread(path, 1)

		if imgNorm == "sub_and_divide":
			img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
		elif imgNorm == "sub_mean":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img[:,:,0] -= 103.939
			img[:,:,1] -= 116.779
			img[:,:,2] -= 123.68
		elif imgNorm == "divide":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img = img/255.0

		img = np.rollaxis(img, 2, 0)
		return img
	except Exception, e:
		print path , e
		img = np.zeros((  height , width  , 3 ))
		img = np.rollaxis(img, 2, 0)
		return img





def getSegmentationArr( path , nClasses ,  width , height  ):

	seg_labels = np.zeros((  height , width  , nClasses ))
	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, ( width , height ))
		img = img[:, : , 0]

		for c in range(nClasses):
			seg_labels[: , : , c ] = (img == c ).astype(int)

	except Exception, e:
		print e
		
	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels



def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):
	
	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'

	images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
	images.sort()
	segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
	segmentations.sort()

	assert len( images ) == len(segmentations)
	for im , seg in zip(images,segmentations):
		assert(  im.split('/')[-1] ==  seg.split('/')[-1] )

	zipped = itertools.cycle( zip(images,segmentations) )

	while True:
		X = []
		Y = []
		for _ in range( batch_size) :
			im , seg = zipped.next()
			X.append( getImageArr(im , input_width , input_height )  )
			Y.append( getSegmentationArr( seg , n_classes , output_width , output_height )  )

		yield np.array(X) , np.array(Y)


#  imageSegmentationGenerator( "data/CamVid/train/" ,  "data/CamVid/trainannot/" ,  128,  10 , 360 , 480 , 300 , 400   ).next()
	
# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer=None , input_image_size=( 360 , 480 )  )




