import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_coef, iou


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def create_dir_train_segmentation(path):
    """
    :param path: Путь к модели
    :return: Создает путь к модели
    """
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling_train_segmentation(x, y):
    """
    :param x: x массив
    :param y: y массив
    :return: Перемешанные x и y массивы
    """
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data_train_segmentation(dataset_path, split=0.1):
    """
    :param dataset_path: Путь к изображениям и маскам
    :param split: коэф. деления на обучающую, валидационную и тестовую выборку
    :return: Массивы значений x и ответы y (Обучающая, валидационная и тестовая выборка)
    """
    images = sorted(glob(os.path.join(dataset_path, "HAM10000_img", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "HAM10000_segmentations_lesion_tschandl", "*.png")))

    test_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image_train_segmentation(path):
    """
    :param path: Путь к изображению
    :return: Нормализованное, измененное по размеру изображение
    """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x                                ## (256, 256, 3)

def read_mask_train_segmentation(path):
    """
    Вход: размеры маски изображения

    :param path: Путь к маске изображения
    :return: Нормализованная матрица маски изображения
    """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)                    ## (256, 256)
    x = np.expand_dims(x, axis=-1)              ## (256, 256, 1)
    return x

def tf_parse_train_segmentation(x, y):
    """
    :param x: Перемешанный массив x
    :param y: Перемешанный массив y
    :return: Перемешанные x и y массивы
    """
    def _parse(x, y):
        x = read_image_train_segmentation(x)
        y = read_mask_train_segmentation(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset_train_segmentation(X, Y, batch):
    """
    :param X: Обучающие массивы x
    :param Y: Обучающий массив y
    :param batch: Размер партии
    :return: обучающий датасет разбитый по партиям
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse_train_segmentation)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset
def dataset_path_train_segmentation(batch_size,lr,num_epochs,Height,Width,dataset_path):
    """
    :param batch_size: размер партии
    :param lr: скорость обучения
    :param num_epochs: кол-во эпох
    :param Height: Высота изображения
    :param Width: Длина изображения
    :param dataset_path: Путь к изображениям и маскам
    :return: обученная модель сегментации
    """
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Folder for saving data """
    create_dir_train_segmentation("files")

    """ Hyperparameters """
    model_path = "files/model.h5"
    csv_path = "files/data.csv"
    global H
    global W
    H = Height
    W = Width
    """ Dataset : 60/20/20 """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data_train_segmentation(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset_train_segmentation(train_x, train_y, batch_size)
    valid_dataset = tf_dataset_train_segmentation(valid_x, valid_y, batch_size)

    train_steps = len(train_x)//batch_size
    valid_steps = len(valid_x)//batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    """ Model """
    model = build_unet((H, W, 3))
    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr), metrics=metrics)
    model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )