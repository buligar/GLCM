import numpy as np
import tensorflow as tf
from keras.layers import Flatten


smooth = 1e-15

def iou(y_true, y_pred):
    """
    :param y_true: Ориганальное значение
    :param y_pred: Предсказанное значение
    :return: Пересечение над объединением (IoU) — это метрика оценки, используемая для измерения точности детектора объектов в конкретном наборе данных.
    """
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    """
    :param y_true: Ориганальное значение
    :param y_pred: Предсказанное значение
    :return: Коэффициент кости между двумя булевыми массивами NumPy или массивоподобными данными.
    """
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    """
    :param y_true: Ориганальное значение
    :param y_pred: Предсказанное значение
    :return: Ошибка кости между двумя булевыми массивами NumPy или массивоподобными данными.
    """
    return 1.0 - dice_coef(y_true, y_pred)