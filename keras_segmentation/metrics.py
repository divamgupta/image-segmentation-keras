import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy

eps = 1e-12


def get_iou(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl) * (pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection) / (union + eps)
        class_wise[cl] = iou
    return class_wise


def indice_jaccard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    interseccion = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    resultado = (interseccion + 1.0) / (union - interseccion + 1.0)

    return K.mean(resultado)


def ternaus_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) - K.log(indice_jaccard(y_true, y_pred))

    return loss


def loss_jaccard(y_true, y_pred):

    return -indice_jaccard(y_true, y_pred)
