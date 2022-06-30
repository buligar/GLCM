import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from keras.utils.generic_utils import custom_object_scope
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_data_train_segmentation, create_dir_train_segmentation


def read_image_test_segmentation(path):
    """
    Вход: Используемые размеры изображения

    :param path: Путь к изображению
    :return: Измененное по размеру изображение, нормализованная матрица пикселей изобарежения
    """
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 256, 256, 3)


def read_mask_test_segmentation(path):
    """
    Вход: Используемые размеры маски изображения

    :param path: Путь к маске изображения
    :return: Измененное по размеру маска изображения, нормализованная матрица пикселей маски изображения
    """
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)                    ## (256, 256)
    return ori_x, x,

def save_results_test_segmentation(ori_x, ori_y, y_pred, save_image_path):
    """
    Вход: Используемые размеры маски изображения

    :param ori_x: Измененное по размеру маска изображения
    :param ori_y: Ориигинальная маска
    :param y_pred: Предсказанная матрица пикселей
    :param save_image_path: # Путь сохранения сравнения результатов сегнментации
    :return: Изображения сравнения результатов сегнментации
    """
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)  ## (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def save_mask_test_segmentation(y_pred, save_image_path):
    """
    :param y_pred: Предсказанная матрица пикселей
    :param save_image_path: Путь сохранения предсказанной маски
    :return: Предсказанные маски изображений
    """
    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    cat_images = np.concatenate([y_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)


def dataset_path_test_segmentation(Height,Width,dataset_path):
    """
    :param Height: Высота
    :param Width: Ширина
    :param dataset_path: Путь к датасету с изображениями и масками
    :return: Метрики результатов сегментации
    """
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    global H
    global W
    H = Height
    W = Width
    """ Folder for saving results """
    create_dir_train_segmentation("results")
    create_dir_train_segmentation("masks")

    """ Load the model """
    with custom_object_scope({'iou': iou, 'dice_coef': dice_coef}):
        model = load_model("files/model.h5")

    """ Load the test data """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data_train_segmentation(dataset_path)

    SCORE = []

    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Exctracting the image name """
        name = x.split("\\")[-1]

        """ Read the image and mask """
        ori_x, x = read_image_test_segmentation(x)
        ori_y, y = read_mask_test_segmentation(y)

        """ Predicting the mask """
        y_pred = model.predict(x)[0] > 0.5
        print(y_pred)
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)

        """ Saving the predicted mask """
        save_image_path = f"results/{name}"
        save_results_test_segmentation(ori_x, ori_y, y_pred, save_image_path)

        """ Saving the mask """
        save_image_path = f"masks/{name}"
        save_mask_test_segmentation(y_pred, save_image_path)

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ mean metrics values """
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns = ["Image Name", "Acc", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv")