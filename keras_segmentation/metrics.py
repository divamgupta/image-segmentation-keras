import numpy as np
from tensorflow.keras import backend as K

EPS = 1e-12


def get_iou(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise

def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return  (2. * intersection + EPS) / (K.sum(y_true_f) + K.sum(y_pred_f) + EPS)

# def dice_coef(y_true, y_pre, smooth=1):
#     # Dice Coefficient is 2 * the Area of Overlap divided by the total number of pixels in both images
#     y_true = np.asarray(y_true, 'bool')
#     y_pre = np.asarray(y_pre, 'bool')
#     inter = np.sum(y_true * y_pre)
#     uni = np.sum(y_true) + np.sum(y_pre)
#
#     return (2 * inter + smooth) / (uni + smooth)


def jacard_index(dice):
    return dice / (2 - dice)
