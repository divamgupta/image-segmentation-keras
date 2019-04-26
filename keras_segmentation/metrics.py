import numpy as np

EPS = 1e-12

def get_iou( gt , pr , n_classes ):
	class_wise = np.zeros(n_classes)
	for cl in range(n_classes):
		intersection = np.sum(( gt == cl )*( pr == cl ))
		union = np.sum(np.maximum( ( gt == cl ) , ( pr == cl ) ))
		iou = float(intersection)/( union + EPS )
		class_wise[ cl ] = iou
	return class_wise
