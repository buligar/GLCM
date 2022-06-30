import sqlite3 # модуль SQlite
import time # библиотека времени
import cv2 # библиотека компьютерного зрения
import numpy # расширение языка Python, добавляющее поддержку больших многомерных массивов и матриц, вместе с большой библиотекой высокоуровневых математических функций для операций с этими массивами.
import matplotlib # библиотека для визуализации данных двумерной графикой
import os # предоставляет функции для взаимодействия с операционной системой
import joblib # это набор инструментов для упрощенной конвейерной обработки в Python
import np as np # все модули numpy и можете использовать их как np.
import numpy as np # все модули numpy и можете использовать их как np.
import pandas as pd # популярный инструментарий анализа данных
import seaborn as sn # библиотека для визуализации
import matplotlib.pylab as plt # набор функций командного стиля, которые заставляют matplotlib работать как MATLAB.
import tensorflow as tf # библиотека Tensorflow для нейросетей
import sys # Не удалять!!!
import sklearn.utils._typedefs # Не удалять !!!
import sklearn.neighbors._partition_nodes # Не удалять !!!
import sklearn.utils._heap # Не удалять !!!
import sklearn.utils._sorting # Не удалять !!!
import sklearn.utils._vector_sentinel # Не удалять !!!
import keras.engine.base_layer_v1 # Не удалять !!!
from keras.utils.generic_utils import custom_object_scope # настраиваемая метрика
from keras.models import load_model # загрузка модели
from PyQt5 import QtCore, QtWidgets, uic # QTCore - Модуль содержит основные классы, в том числе цикл событий и механизм сигналов и слотов Qt. Вспомогательные модули, для работы с виджетами и ui-фалйами, сгенерированными в дизайнере
from PyQt5.QtWidgets import QMenuBar, QMenu, QFileDialog,QGraphicsScene,QRadioButton  # подключение виджетов
from PyQt5.QtGui import QPixmap,QIcon # подключение модуля для работы с изображениями
from skimage.feature import greycomatrix, greycoprops# подключение модуля для построения МПС и вычисления текстурных признаков
from matplotlib import pyplot as plt # Pyplot предоставляет интерфейс конечного автомата для базовой библиотеки построения графиков в matplotlib.
from tkinter import * # пакет для Python, предназначенный для работы с библиотекой Tk
from sklearn.metrics import precision_recall_fscore_support as score # метрики
from sklearn.model_selection import cross_val_score, train_test_split # Оценка баллов перекрестной проверкой и разделение на тренировочную и тестовую выборку
from sklearn import preprocessing # предоставляет несколько общих служебных функций и классов преобразования для преобразования необработанных векторов признаков в представление, более подходящее для последующих оценок.
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression # логистическая регрессия
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # линейный дискриминантный анализ
from sklearn.neighbors import KNeighborsClassifier # к-ближайших соседей
from sklearn.tree import DecisionTreeClassifier # деревья решений
from sklearn.naive_bayes import GaussianNB # Наивный Байес
from sklearn.svm import SVC # метод опорных векторов
from sklearn.neural_network import MLPClassifier # Многослойный перцептрон
from sklearn.model_selection import KFold # К-проверка
from metrics import dice_coef, iou # функции потери коэффициента и индексом Жаккара, по сути является методом количественной оценки процентного перекрытия между целевой маской и нашим прогнозируемым результатом.
from train import dataset_path_train_segmentation, create_dir_train_segmentation # подключение функций создания путей и обучения сегментации
from eval import dataset_path_test_segmentation # подключение проверки модели сегментации


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # INFO и WARNING сообщения не печатаются
matplotlib.use('QT5Agg') # подключение бэкенда QTAgg, представляет собой неинтерактивный бэкенд, который может записывать только в файлы


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self): # Инициализация
        """
        Инициализация интерфейса, импорт интерфейса, вызов функций
        """
        super(MyWindow, self).__init__()
        self.ui=uic.loadUi('GLCM.ui', self) # Импорт интерфейса
        self.setWindowTitle("Система бинарной классификации изображений по текстурным признакам")
        self.setWindowIcon(QIcon('logo.png'))
        self.addFunctions() # Вызов функций

    # Декоратор подсчета времени выполнения функции
    def timer(func):
        """
        Вход: Подается функция

        :return: время работы функции
        """
        def wrapper(*args, **kwargs):
            start = time.time()
            val = func(*args, **kwargs)
            print(f"{time.time()-start}")
            return val
        return wrapper

    def addFunctions(self):
        """
        Подключение функций взаимодействия с интерфейсом

        :return: None
        """
        fileMenu = QMenu("&Файл", self)
        fileMenu.addAction('Открыть', self.action_clicked)
        fileMenu.addAction('Загрузить выборку', self.action_clicked)
        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)
        self.menuBar.addMenu(fileMenu)
        self.database = f'{self.ui.database_name.text()}.db'
        self.table = 'one'
        self.listWidget.itemClicked.connect(self.listitemclicked) # Загрузка изображений в список
        self.pushButton_deletehair.clicked.connect(lambda: self.removehair()) # Удаление волос
        self.pushButton.clicked.connect(lambda: self.glcm_one(k=0)) # GLCM
        self.pushButton_2.clicked.connect(lambda: self.glcm_all()) # GLCM обучающая и тестовая выборка
        self.info_button.clicked.connect(lambda: self.informativ()) # Информативность в общем
        self.zoom_in_button.clicked.connect(lambda: self.on_zoom_in()) # Увеличить src
        self.zoom_out_button.clicked.connect(lambda: self.on_zoom_out()) # Уменьшить src
        self.zoom_in_button_2.clicked.connect(lambda: self.on_zoom_in2())  # Увеличить dst
        self.zoom_out_button_2.clicked.connect(lambda: self.on_zoom_out2())  # Уменьшить dst
        self.pushButton_segmentation_2.clicked.connect(lambda: self.segmentation_Unet()) # Сегментация U-net
        self.pushButton_train_segmentation_U_net.clicked.connect(lambda: self.train_segmentation_Unet()) # Обучение U-net
        self.pushButton_test_segmentation_U_net.clicked.connect(lambda: self.test_segmentation_Unet()) # Тестирование U-net
        self.pushButton_delete_items.clicked.connect(lambda: self.deleteitems()) # удаление всех элементов из списка
        self.pushButton_delete_item.clicked.connect(lambda: self.deleteitem()) # удаление элемента из списка
        self.MLA_button.clicked.connect(lambda: self.MLA()) # Мульти-классификация

    @QtCore.pyqtSlot()
    def action_clicked(self):  # Строка меню
        """
        Вход: Сигналы с кнопок открыть, загрузить выборку, загрузить метаданные

        :return: Пути выбранных папок, файлов
        """
        action = self.sender()
        if action.text() == "Открыть":
            try:
                self.files_path = QFileDialog.getOpenFileNames(self)[0]  # извлечение изображений из Dialog окна
                print(self.files_path)
                self.listWidget.addItems(self.files_path)  # добавление изображений в список
            except FileNotFoundError:
                print("No such file")
        elif action.text() == "Загрузить выборку":
            try:
                self.train_papka=QFileDialog.getExistingDirectory(self) # Путь к папке содержащей папки классов
                print(self.train_papka)
            except NotADirectoryError:
                print("No such directory")


    def listitemclicked(self): # Выбор из списка изображения и отображение на label
        """
        Вход: список изображений

        :return: отображение изображения на src и dst
        """

        self.file_path = self.listWidget.selectedItems()[0].text()  # Путь к выбранному файлу
        # 1 окно
        self.scene = QGraphicsScene(self.graphicsView)
        self.pixmap = QPixmap(self.file_path)
        self.scene.addPixmap(self.pixmap)
        self.graphicsView.setScene(self.scene)
        # 2 окно
        self.scene_2 = QGraphicsScene(self.graphicsView_2)
        self.pixmap2 = QPixmap(self.file_path)
        self.scene_2.addPixmap(self.pixmap2)
        self.graphicsView_2.setScene(self.scene_2)

        self.scale = 1

    def deleteitem(self): # Удаление элемента из списка
        """
        Вход: список изображений

        :return: удаление выбранного изображения
        """
        listItems = self.listWidget.selectedItems()
        if not listItems: return
        for item in listItems:
            self.listWidget.takeItem(self.listWidget.row(item))
        self.scene.clear()
        self.scene_2.clear()

    def deleteitems(self): # Удаление элементов из списка
        """
        Вход: список изображений, src сцена, dst сцена

        :return: очистка списка и src, dst сцен
        """
        try:
            self.listWidget.clear()
            self.scene.clear()
            self.scene_2.clear()
        except:
            print('Список уже пуст')
    def on_zoom_in(self): # Увеличение изображения 1 окна
        """
        Вход: изображение со сцены src

        :return: увеличенное изображение src
        """
        self.scene.clear()
        self.scale *= 2
        self.resize_image()

    def on_zoom_out(self): # Уменьшение изображения 1 окна
        """
        Вход: изображение со сцены dst

        :return: уменьшенное изображение src
        """
        self.scene.clear()
        self.scale /= 2
        self.resize_image()

    def on_zoom_in2(self): # Увеличение изображения 2 окна
        """
        Вход: изображение со сцены dst

        :return: увеличенное изображение dst
        """
        self.scene_2.clear()
        self.scale *= 2
        self.resize_image2()

    def on_zoom_out2(self): # Уменьшение изображения 2 окна
        """
        Вход: изображение со сцены dst

        :return: уменьшенное изображение dst
        """
        self.scene_2.clear()
        self.scale /= 2
        self.resize_image2()

    def resize_image(self): # Изменение изображения 1 окна
        """
        Вход: Pixmap изображения, масштаб

        :return: измененное изображение на src
        """
        size = self.pixmap.size()
        scaled_pixmap = self.pixmap.scaled(self.scale * size)
        self.scene.addPixmap(scaled_pixmap)

    def resize_image2(self): # Изменение изображения 2 окна
        """
        Вход: Pixmap изображения, масштаб

        :return: измененное изображение на dst
        """
        size = self.pixmap2.size()
        scaled_pixmap = self.pixmap2.scaled(self.scale * size)
        self.scene_2.addPixmap(scaled_pixmap)

    def removehair(self):
        """
        Вход: Путь к изображению

        :return: Измененное изображение без волос
        """
        image = cv2.imread(self.file_path)
        height, width = image.shape[:2]
        image_resize = cv2.resize(image, (width, height))
        grayScale = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY) # Преобразование исходного изображения в оттенки серого
        kernel = cv2.getStructuringElement(1, (17, 17)) # Ядро для морфологической фильтрации
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) # Выполните фильтрацию черной шляпы на изображении в градациях серого, чтобы найти контуры волос.
        ret, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY) # усилить контуры волос при подготовке к окрашиванию
        final_image = cv2.inpaint(image_resize, threshold, 1, cv2.INPAINT_TELEA) # закрасить исходное изображение в зависимости от маски
        cv2.imwrite("dst.png", final_image)
        self.pixmap2 = QPixmap('dst.png')
        self.scene_2 = QGraphicsScene(self.graphicsView_2)
        self.scene_2.addPixmap(self.pixmap2)
        self.graphicsView_2.setScene(self.scene_2)
        self.file_path = 'dst.png'
        self.dst_path = 'dst.png'


    @timer
    def train_segmentation_Unet(self): # Обучение сегментации U-net
        """
        Вход: Путь к папке с изображениями и масками, размеры изображения модели U-net (кратное 2, высота и ширина одинаковы),
        размер партии, скорость обучения, кол-во эпох

        :return: обученная модель сегментации U-net
        """
        # dataset_path="D:/Downloads/ISIC2018/"
        dataset_path="input\\HAM10000\\"
        Height = 256
        Width = 256
        batch_size = 4
        lr = 0.0001
        num_epochs = 30
        dataset_path_train_segmentation(batch_size,lr,num_epochs,Height,Width,dataset_path)
        print("Время обучения сегментации:")

    def test_segmentation_Unet(self): # Тестирование сегмментации U-net
        """
        Вход: Путь к папке с изображениями и масками, размеры изображения модели U-net

        :return: Результаты метрик сегментирования
        """
        # dataset_path = "D:\\Downloads\\ISIC2018\\"
        dataset_path="input\\HAM10000\\"
        Height = 256
        Width = 256
        dataset_path_test_segmentation(Height,Width,dataset_path)


    def segmentation_Unet(self): # Сегментация c загруженной моделью нейронной сети U-net
        """
        Вход: Размеры изображения модели U-net, путь к файлу

        :return: Сегментированное изображение
        """
        np.set_printoptions(edgeitems=100000)
        np.random.seed(42)
        tf.random.set_seed(42)
        create_dir_train_segmentation("masks") # Создание папки masks
        H = 256
        W = 256
        with custom_object_scope({'iou': iou, 'dice_coef': dice_coef}):  # Загрузка модели
            model = load_model("files/model.h5")

        # Загрузка изображения
        image = cv2.imread(self.file_path, cv2.IMREAD_COLOR)  # Загрузка изображения
        height, width = image.shape[:2]
        img = cv2.resize(image, (W, H))  # Изменить размер
        image_resize = np.array(image, dtype=np.uint8)  # Преобразовать в массив
        x = img / 255.0  # Нормализация [0,1]
        x = x.astype(np.float32)  # Преобразовать во float
        x = np.expand_dims(x, axis=0)

        # Прогнозирование маски
        y_pred = model.predict(x)[0] > 0.5  # Предугадать по модели
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.uint8)
        y_pred = cv2.resize(y_pred,(width,height))
        # cv2.imwrite("mask.png", y_pred)
        # self.file_path = 'mask.png'
        # self.filters()
        # y_pred = cv2.imread('blurred.png',0)
        dst = cv2.bitwise_and(image_resize, image_resize, mask=y_pred)

        # Вывод на экран
        cv2.imwrite("dst.png", dst)
        self.scene_2 = QGraphicsScene(self.graphicsView_2)
        self.pixmap2 = QPixmap('dst.png')
        self.scene_2.addPixmap(self.pixmap2)
        self.graphicsView_2.setScene(self.scene_2)
        self.file_path = 'dst.png'
        self.dst_path = 'dst.png'


    def glcm_vn(self,grayscale):
        """
        Вход: режим отображения, сохранения изображений и текстурных признаков; цвет канала; расстояние смежности

        :param grayscale: глобальные переменные self и матрица уровней серого grayscale
        :return: матрица текстурных признаков
        """
        radioButton = self.ui.save_plot_features.currentIndex()
        if self.tsvet == 0:
            grayscale[:, :, 0] = 0
            grayscale[:, :, 1] = 0
            if radioButton == 1:
                cv2.imwrite('r.bmp', grayscale)
                self.scene_2 = QGraphicsScene(self.graphicsView_2)
                self.pixmap2 = QPixmap('r.bmp')
                self.scene_2.addPixmap(self.pixmap2)
                self.graphicsView_2.setScene(self.scene_2)
                cv2.waitKey(1)
        elif self.tsvet == 1:
            grayscale[:, :, 0] = 0
            grayscale[:, :, 2] = 0
            if radioButton == 1:
                cv2.imwrite('g.bmp', grayscale)
                self.scene_2 = QGraphicsScene(self.graphicsView_2)
                self.pixmap2 = QPixmap('g.bmp')
                self.scene_2.addPixmap(self.pixmap2)
                self.graphicsView_2.setScene(self.scene_2)
                cv2.waitKey(1)
        elif self.tsvet == 2:
            grayscale[:, :, 1] = 0
            grayscale[:, :, 2] = 0
            if radioButton == 1:
                cv2.imwrite('b.bmp', grayscale)
                self.scene_2 = QGraphicsScene(self.graphicsView_2)
                self.pixmap2 = QPixmap('b.bmp')
                self.scene_2.addPixmap(self.pixmap2)
                self.graphicsView_2.setScene(self.scene_2)
                cv2.waitKey(1)
        elif self.tsvet == 3:
            grayscale = cv2.imread('dst.png', 0)
            if radioButton == 1:
                cv2.imwrite('gray.bmp', grayscale)
                self.scene_2 = QGraphicsScene(self.graphicsView_2)
                self.pixmap2 = QPixmap('gray.bmp')
                self.scene_2.addPixmap(self.pixmap2)
                self.graphicsView_2.setScene(self.scene_2)
                cv2.waitKey(1)

            # Извлечение канала цвета
        # grayscale = cv2.imread(self.file_path)  # Преобразование в оттенки серого
        if self.tsvet == 0:
            grayscale = grayscale[:, :, 2]
        elif self.tsvet == 1:
            grayscale = grayscale[:, :, 1]
        elif self.tsvet == 2:
            grayscale = grayscale[:, :, 0]
        elif self.tsvet == 3:
            grayscale = cv2.imread(self.file_path, 0)

        glcm = greycomatrix(grayscale, distances=self.Distances,
                            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=256,
                            symmetric=True, normed=True)  # Построение МПС


        # print(glcm)
        filt_glcm = glcm[1:, 1:, :, :]  # Не берет в расчет пиксель 0

        self.Contrast = greycoprops(filt_glcm, 'contrast')  # Текстурный признак Контраст
        self.Dissimilarity = greycoprops(filt_glcm, 'dissimilarity')  # Текстурный признак несходство
        self.Homogeneity = greycoprops(filt_glcm, 'homogeneity')  # Текстурный признак Локальная однородность
        self.Asm = greycoprops(filt_glcm, 'ASM')  # Текстурный признак Угловой второй момент
        self.Energy = greycoprops(filt_glcm, 'energy')  # Текстурный признак Энергия
        self.Correlation = greycoprops(filt_glcm, 'correlation')  # Текстурный признак Корреляция
        self.Entropy = greycoprops(filt_glcm, 'entropy') # Текстурный признак Энтропия
        self.Max = greycoprops(filt_glcm,'MAX') # Текстурный признак Максимум вероятности

        # из двумерного массива в одномерный
        Contrast = np.concatenate(self.Contrast)
        Dissimilarity = np.concatenate(self.Dissimilarity)
        Homogeneity = np.concatenate(self.Homogeneity)
        Asm = np.concatenate(self.Asm)
        Energy = np.concatenate(self.Energy)
        Correlation = np.concatenate(self.Correlation)
        Entropy = np.concatenate(self.Entropy)
        Max = np.concatenate(self.Max)


        if radioButton == 1:
            Contrast_all = np.transpose(self.Contrast)
            Contrast0 = Contrast_all[0]
            Contrast45 = Contrast_all[1]
            Contrast90 = Contrast_all[2]
            Contrast135 = Contrast_all[3]
            with open('texture\Contrast0.csv', 'a') as f:
                np.savetxt(f, Contrast0, delimiter=';', fmt='%.5f')
            with open('texture\Contrast45.csv', 'a') as f:
                np.savetxt(f, Contrast45, delimiter=';', fmt='%.5f')
            with open('texture\Contrast90.csv', 'a') as f:
                np.savetxt(f, Contrast90, delimiter=';', fmt='%.5f')
            with open('texture\Contrast135.csv', 'a') as f:
                np.savetxt(f, Contrast135, delimiter=';', fmt='%.5f')

            Dissimilarity_all = np.transpose(self.Dissimilarity)
            Dissimilarity0 = Dissimilarity_all[0]
            Dissimilarity45 = Dissimilarity_all[1]
            Dissimilarity90 = Dissimilarity_all[2]
            Dissimilarity135 = Dissimilarity_all[3]
            with open('texture\Dissimilarity0.csv', 'a') as f:
                np.savetxt(f, Dissimilarity0, delimiter=';', fmt='%.5f')
            with open('texture\Dissimilarity45.csv', 'a') as f:
                np.savetxt(f, Dissimilarity45, delimiter=';', fmt='%.5f')
            with open('texture\Dissimilarity90.csv', 'a') as f:
                np.savetxt(f, Dissimilarity90, delimiter=';', fmt='%.5f')
            with open('texture\Dissimilarity135.csv', 'a') as f:
                np.savetxt(f, Dissimilarity135, delimiter=';', fmt='%.5f')

            Homogeneity_all = np.transpose(self.Homogeneity)
            Homogeneity0 = Homogeneity_all[0]
            Homogeneity45 = Homogeneity_all[1]
            Homogeneity90 = Homogeneity_all[2]
            Homogeneity135 = Homogeneity_all[3]
            with open('texture\Homogeneity0.csv', 'a') as f:
                np.savetxt(f, Homogeneity0, delimiter=';', fmt='%.5f')
            with open('texture\Homogeneity45.csv', 'a') as f:
                np.savetxt(f, Homogeneity45, delimiter=';', fmt='%.5f')
            with open('texture\Homogeneity90.csv', 'a') as f:
                np.savetxt(f, Homogeneity90, delimiter=';', fmt='%.5f')
            with open('texture\Homogeneity135.csv', 'a') as f:
                np.savetxt(f, Homogeneity135, delimiter=';', fmt='%.5f')

            Asm_all = np.transpose(self.Asm)
            Asm0 = Asm_all[0]
            Asm45 = Asm_all[1]
            Asm90 = Asm_all[2]
            Asm135 = Asm_all[3]
            with open('texture\Asm0.csv', 'a') as f:
                np.savetxt(f, Asm0, delimiter=';', fmt='%.5f')
            with open('texture\Asm45.csv', 'a') as f:
                np.savetxt(f, Asm45, delimiter=';', fmt='%.5f')
            with open('texture\Asm90.csv', 'a') as f:
                np.savetxt(f, Asm90, delimiter=';', fmt='%.5f')
            with open('texture\Asm135.csv', 'a') as f:
                np.savetxt(f, Asm135, delimiter=';', fmt='%.5f')

            Energy_all = np.transpose(self.Energy)
            Energy0 = Energy_all[0]
            Energy45 = Energy_all[1]
            Energy90 = Energy_all[2]
            Energy135 = Energy_all[3]
            with open('texture\Energy0.csv', 'a') as f:
                np.savetxt(f, Energy0, delimiter=';', fmt='%.5f')
            with open('texture\Energy45.csv', 'a') as f:
                np.savetxt(f, Energy45, delimiter=';', fmt='%.5f')
            with open('texture\Energy90.csv', 'a') as f:
                np.savetxt(f, Energy90, delimiter=';', fmt='%.5f')
            with open('texture\Energy135.csv', 'a') as f:
                np.savetxt(f, Energy135, delimiter=';', fmt='%.5f')

            Correlation_all = np.transpose(self.Correlation)
            Correlation0 = Correlation_all[0]
            Correlation45 = Correlation_all[1]
            Correlation90 = Correlation_all[2]
            Correlation135 = Correlation_all[3]
            with open('texture\Correlation0.csv', 'a') as f:
                np.savetxt(f, Correlation0, delimiter=';', fmt='%.5f')
            with open('texture\Correlation45.csv', 'a') as f:
                np.savetxt(f, Correlation45, delimiter=';', fmt='%.5f')
            with open('texture\Correlation90.csv', 'a') as f:
                np.savetxt(f, Correlation90, delimiter=';', fmt='%.5f')
            with open('texture\Correlation135.csv', 'a') as f:
                np.savetxt(f, Correlation135, delimiter=';', fmt='%.5f')

            Entropy_all = np.transpose(self.Entropy)
            Entropy0 = Entropy_all[0]
            Entropy45 = Entropy_all[1]
            Entropy90 = Entropy_all[2]
            Entropy135 = Entropy_all[3]
            with open('texture\Entropy0.csv', 'a') as f:
                np.savetxt(f, Entropy0, delimiter=';', fmt='%.5f')
            with open('texture\Entropy45.csv', 'a') as f:
                np.savetxt(f, Entropy45, delimiter=';', fmt='%.5f')
            with open('texture\Entropy90.csv', 'a') as f:
                np.savetxt(f, Entropy90, delimiter=';', fmt='%.5f')
            with open('texture\Entropy135.csv', 'a') as f:
                np.savetxt(f, Entropy135, delimiter=';', fmt='%.5f')

            Max_all = np.transpose(self.Max)
            Max0 = Max_all[0]
            Max45 = Max_all[1]
            Max90 = Max_all[2]
            Max135 = Max_all[3]
            with open('texture\Max0.csv', 'a') as f:
                np.savetxt(f, Max0, delimiter=';', fmt='%.5f')
            with open('texture\Max45.csv', 'a') as f:
                np.savetxt(f, Max45, delimiter=';', fmt='%.5f')
            with open('texture\Max90.csv', 'a') as f:
                np.savetxt(f, Max90, delimiter=';', fmt='%.5f')
            with open('texture\Max135.csv', 'a') as f:
                np.savetxt(f, Max135, delimiter=';', fmt='%.5f')
            self.plot_features()

        # Сохранение все вместе
        self.GLCM_All = [Contrast] + [Dissimilarity] + [Homogeneity] + [Asm] + [Energy] + [Correlation] + [Entropy] + [Max]
        self.GLCM_All = np.concatenate(self.GLCM_All)
        self.GLCM_RGB.append(self.GLCM_All)

    def import_src_binary(self): # получаем бинарный код изображения
        """
        Вход: Путь к изображению

        :return: бинарный код изображения src
        """
        f = open(self.file_path,'rb')
        src_binary = f.read()
        return src_binary
    def import_dst_binary(self): # получаем бинарный код изображения
        """
        Вход: Путь к изображению

        :return: бинарный код изображения dst
        """
        f = open(self.dst_path,'rb')
        dst_binary = f.read()
        return dst_binary
    def sqlite3_simple_pict_import(self,i,k): # импорт изображений в базу данных
        """
        Вход: Путь к папке для создания базы данных, название таблицы, бинарный код с изображений src и dst

        :param i: Номер изображения
        :param k: Номер класса
        :return: Созданная база данных изображений
        """
        con = sqlite3.connect(database=self.database) # соединение с базой данны
        cur = con.cursor() # создаем объект курсора
        query_connection = 'CREATE TABLE IF NOT EXISTS '+str(self.table)+' (id TEXT, class TEXT, path TEXT, src BLOB, dst BLOB)'
        # создаем таблицу если ее не существует
        cur.execute(query_connection)
        binary_pict_src = self.import_src_binary()
        binary_pict_dst = self.import_dst_binary()
        data = (str(i),str(k),self.file_path,binary_pict_src,binary_pict_dst)
        query = 'INSERT INTO '+self.table+' VALUES(?, ?, ?, ?, ?)'
        # выполяем запрос вставки данных
        cur.execute(query,data)
        con.commit()
        cur.close()
        con.close()

    def plot_features(self): #Отображение на графиках
        """
        Вход: строка расстояний смежности, расстояние смежности, текстурные признаки

        :return: график отображения значений текстурных признаков
        """
        plt.subplot(2, 4, 1)
        plt.grid(axis='both')
        plt.title("Контраст")
        plt.xticks([i for i in range(0, max(self.arrayDistances + 1),20)])
        plt.plot(self.Distances, self.Contrast, marker='o')

        plt.subplot(2, 4, 2)
        plt.grid(axis='both')
        plt.title("Несходство")
        plt.xticks([i for i in range(0, max(self.arrayDistances + 1),20)])
        plt.plot(self.Distances, self.Dissimilarity, marker='o')

        plt.subplot(2, 4, 3)
        plt.grid(axis='both')
        plt.title("Локальная однородность")
        plt.xticks([i for i in range(0, max(self.arrayDistances + 1),20)])
        plt.plot(self.Distances, self.Homogeneity, marker='o')

        plt.subplot(2, 4, 4)
        plt.grid(axis='both')
        plt.title("Угловой второй момент")
        plt.xticks([i for i in range(0, max(self.arrayDistances + 1),20)])
        plt.plot(self.Distances, self.Asm, marker='o')

        plt.subplot(2, 4, 5)
        plt.grid()
        plt.title("Энергия")
        plt.xticks([i for i in range(0, max(self.arrayDistances + 1),20)])
        plt.plot(self.Distances, self.Energy, marker='o')

        plt.subplot(2, 4, 6)
        plt.grid(axis='both')
        plt.title("Корреляция")
        plt.xticks([i for i in range(0, max(self.arrayDistances + 1),20)])
        plt.plot(self.Distances, self.Correlation, marker='o')

        plt.subplot(2, 4, 7)
        plt.grid(axis='both')
        plt.title("Энтропия")
        plt.xticks([i for i in range(0, max(self.arrayDistances + 1),20)])
        plt.plot(self.Distances, self.Entropy, marker='o')

        plt.subplot(2, 4, 8)
        plt.grid(axis='both')
        plt.title("Максимум вероятности")
        plt.xticks([i for i in range(0, max(self.arrayDistances + 1),20)])
        plt.plot(self.Distances, self.Max, marker='o')

        self.Anglelegend = [0, 45, 90, 135]
        plt.figlegend(self.Anglelegend)
        plt.show()

    @timer
    def glcm_one(self,k):
        """
        :param k: номер класса
        Вход: режим расчета МПС, режим сегментации, цвет канала, расстояние смежности, список путей изображений

        :return: файл glcm_all.csv - текстурные признаки изображений
        """
        # Очищение файла glcm_all.csv
        my_file = open("csv/glcm_all.csv", "w+")
        my_file.close()
        file_csv_path = 'csv/glcm_all.csv'
        os.remove(file_csv_path)
        mode = self.ui.modeBox.currentIndex()
        color = self.ui.color.currentIndex()
        if mode == 0:
            if color == 4:
                self.features_sum = 8 * 4 * 4
                head = np.array(range(1, self.features_sum + 1)).flatten()
                self.Distances = [self.spinBox.value()]
                value = self.spinBox.value()
                self.arrayDistances = np.arange(1, value + 1, 1)
            else:
                self.features_sum = 8 * 4
                head = np.array(range(1, self.features_sum + 1)).flatten()
                self.Distances = [self.spinBox.value()]
                value = self.spinBox.value()
                self.arrayDistances = np.arange(1, value + 1, 1)
        elif mode == 1:
            if color == 4:
                self.features_sum = 8 * 4 * 4 * self.spinBox.value()
                head = np.array(range(1, self.features_sum + 1)).flatten()
                value = self.spinBox.value()
                self.Distances = np.arange(1, value + 1, 1)
                self.arrayDistances = np.arange(1, value + 1, 1)
            else:
                self.features_sum = 8 * 4 * self.spinBox.value()
                head = np.array(range(1, self.features_sum + 1)).flatten()
                value = self.spinBox.value()
                self.Distances = np.arange(1, value + 1, 1)
                self.arrayDistances = np.arange(1, value + 1, 1)
        num = len(self.files_path) # Всего изображений
        self.num_cells=np.array(range(1,num+1)).flatten() # Массив кол-ва изображений
        df = pd.DataFrame(np.matrix(head)) # Создание заголовков столбцов
        df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';') # Указание кол-ва признаков
        select_segmentation = self.ui.select_segmentation.currentIndex()
        k+=1
        for i in range(num):
            self.file_path=self.files_path[i]
            print(self.file_path)
            self.scene = QGraphicsScene(self.graphicsView)
            self.pixmap = QPixmap(self.file_path)
            self.scene.addPixmap(self.pixmap)
            self.graphicsView.setScene(self.scene)
            if select_segmentation == 0: ### Сегментация U-net
                self.removehair()
                self.segmentation_Unet()
                np.set_printoptions(edgeitems=1000) # для отображения полной матрицы изображения в окне вывода
                self.dst_path = 'dst.png'
                self.sqlite3_simple_pict_import(i, k)
                self.scale = 1
                self.GLCM_RGB = []
                # Сохранение и отображение канала цвета
                if color < 4:
                    self.tsvet = color
                    grayscale = cv2.imread(self.dst_path)
                    self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
                elif color == 4 :
                    for t in range(4):
                        self.tsvet = t
                        grayscale = cv2.imread(self.dst_path)
                        self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
            elif select_segmentation == 1: ### Удаление волос
                self.removehair()
                np.set_printoptions(edgeitems=1000)  # для отображения полной матрицы изображения в окне вывода
                self.dst_path = 'dst.png'
                self.sqlite3_simple_pict_import(i, k)
                self.GLCM_RGB = []
                # Сохранение и отображение канала цвета
                if color < 4:
                    self.tsvet = color
                    grayscale = cv2.imread(self.dst_path)
                    self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
                elif color == 4:
                    for k in range(4):
                        self.tsvet = k
                        grayscale = cv2.imread(self.dst_path)
                        self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
            elif select_segmentation == 2: ### Сегментация
                self.segmentation_Unet()
                np.set_printoptions(edgeitems=1000)  # для отображения полной матрицы изображения в окне вывода
                self.dst_path = 'dst.png'
                self.sqlite3_simple_pict_import(i, k)
                self.GLCM_RGB = []
                # Сохранение и отображение канала цвета
                if color < 4:
                    self.tsvet = color
                    grayscale = cv2.imread(self.dst_path)
                    self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
                elif color == 4:
                    for k in range(4):
                        self.tsvet = k
                        grayscale = cv2.imread(self.dst_path)
                        self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
            elif select_segmentation == 3:
                np.set_printoptions(edgeitems=1000)  # для отображения полной матрицы изображения в окне вывода
                dst = cv2.imread(self.file_path)
                cv2.imwrite('dst.png', dst)
                self.dst_path = 'dst.png'
                self.sqlite3_simple_pict_import(i, k)
                self.GLCM_RGB = []
                # Сохранение и отображение канала цвета
                if color < 4:
                    self.tsvet = color
                    grayscale = cv2.imread(self.dst_path)
                    self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
                elif color == 4:
                    for k in range(4):
                        self.tsvet = k
                        grayscale = cv2.imread(self.dst_path)
                        self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
        print("Время выполнения работы функции расчета признаков с учетом параметров:")

    @timer
    def glcm_all(self):
        """
        Вход: путь к обучающей папке, glcm_all.csv

        :return: train и test .csv файлы разных классов
        """
        path = self.train_papka
        list = os.listdir(self.train_papka)
        print(list)
        for k in range(10):
            my_file = open(f"csv/glcm_train{k}.csv", "w+")
            my_file.close()
            file_csv_path = f'csv/glcm_train{k}.csv'
            os.remove(file_csv_path)
            my_file = open(f"csv/glcm_test{k}.csv", "w+")
            my_file.close()
            file_csv_path = f'csv/glcm_test{k}.csv'
            os.remove(file_csv_path)
        file_count = len(list)
        my_file = open("csv/glcm_all.csv", "w+")
        my_file.close()
        file_csv_path = 'csv/glcm_all.csv'
        os.remove(file_csv_path)
        print(file_count)
        for k in range(file_count):
            papka = []
            for p in os.listdir(path):
                full_path = os.path.join(path, p).replace('/','\\')
                papka += [full_path]
            print(papka)
            self.files_path_all = []
            for l in os.listdir(papka[k]):
                full_path = os.path.join(papka[k], l)
                self.files_path_all += [full_path]
            print(self.files_path_all)
            self.files_path, x_test = train_test_split(self.files_path_all, test_size=0.2, random_state=42)
            if self.ui.database_name.text() == False:
                print("Введите название базы данных")
            else:
                self.database = f'{self.ui.database_name.text()}.db'
                self.table = 'train'
                self.glcm_one(k)
                # self.glcm_test(k)
                file_oldname = os.path.join("csv/glcm_all.csv").replace('\\','/')
                file_newname_newfile = os.path.join(f"csv/glcm_train{k}.csv").replace('\\','/')
                os.rename(file_oldname, file_newname_newfile)

            x_train, self.files_path = train_test_split(self.files_path_all, test_size=0.2, random_state=42)
            if self.ui.database_name.text() == False:
                print("Введите название базы данных")
            else:
                self.database = f'{self.ui.database_name.text()}.db'
                self.table = 'test'
                self.glcm_one(k)
                # self.glcm_test(k)
                file_oldname = os.path.join("csv/glcm_all.csv").replace('\\','/')
                file_newname_newfile = os.path.join(f"csv/glcm_test{k}.csv").replace('\\','/')
                os.rename(file_oldname, file_newname_newfile)
        print("Время выполнения работы расчета признаков обучающей и тестовой выборки:")

    def old_correllation(self):
        """
        Вход: Режим расчета МПС, признаки двух классов, расстояние смежности

        :return: Выбранные некоррелированные признаки
        """
        mode = self.ui.modeBox.currentIndex()
        plt.figure(figsize=(16, 6))
        corr_matrix = self.features.loc[:, :].corr()
        sn.heatmap(corr_matrix)
        plt.show()
        plt.figure(figsize=(16, 6))
        corr_matrix = 0.3 < self.features.loc[:, :].corr() < 0.7
        print("0.9 Корреляционная матрица", corr_matrix)
        max_corr = max(corr_matrix.gt(0).sum(axis=1)) * 90 / 100
        print("Максимум корреляции", max_corr)
        max_corr_matrix = corr_matrix.gt(0).sum(axis=0)
        print(len(np.array(max_corr_matrix)))
        self.rass = self.spinBox.value()
        if mode == 0:
            num = self.rass
        elif mode == 1:
            num = self.rass * 128
        features = np.array(max_corr_matrix)
        print(features)
        self.max_features = []
        for i in range(num):
            if features[i] > max_corr:
                self.max_features.append(i)
        print((np.array(self.max_features)))
        # Using heatmap to visualize the correlation matrix
        sn.heatmap(corr_matrix)
        plt.show()

    def new_features(self):
        """
        Вход: Выбранные признаки (После корреляции или после информативности)

        :return: Признаки двух классов
        """
        features = pd.read_csv('csv/glcm_train0.csv', delimiter=';', usecols=np.array(self.max_features))
        self.mean1 = features.describe().loc[['mean']]
        self.std1 = features.describe().loc[['std']]
        features = pd.read_csv('csv/glcm_train1.csv', delimiter=';', usecols=np.array(self.max_features))
        self.mean2 = features.describe().loc[['mean']]
        self.std2 = features.describe().loc[['std']]
        self.features = pd.concat([features, features])
        self.y_train = self.features.iloc[:, -1:].to_numpy().flatten()

    def infor(self):
        """
        Вход: расстояние смежности, среднее значение первого и второго класса, среднеквадратическое значение первого и второго класса

        :return: Выбранные информативные признаки
        """
        if self.spinBox.value() == 0:
            print("Укажите расстояние смежности в spinbox")
        self.rass = self.spinBox.value()  # Расстояние смежности
        c = numpy.array(abs(self.mean1 - self.mean2))
        z = numpy.array(1.6 * (self.std1 + self.std2))
        self.informativeness = numpy.divide(c, z)
        max_info = np.max(self.informativeness)
        len_info = len(self.informativeness[0])
        self.informativeness = np.array(self.informativeness[0].reshape(len_info, 1))
        max_info = max_info * 50 / 100
        self.max_features = []
        for i in range(len_info):
            if self.informativeness[i] > max_info:
                self.max_features.append(i)
        self.len_max_features = len(self.max_features)
        return self.max_features


    def load2class(self): # Загрузка 2 классов обучающей выборки
        """
        Вход: glcm_train0.csv и glcm_train1.csv текстурные признаки 1 и 2 класса

        :return: массивы средних значений и стандартных отклонений изображений 1 и 2 класса
        """

        my_file = open("csv/max_corr_matrix.csv", "w+")
        my_file.close()
        file_csv_path = 'csv/max_corr_matrix.csv'
        os.remove(file_csv_path)
        np.set_printoptions(edgeitems=100000)
        features = pd.read_csv('csv/glcm_train0.csv', delimiter=';')
        self.mean1 = features.describe().loc[['mean']]
        self.std1 = features.describe().loc[['std']]
        features = pd.read_csv('csv/glcm_train1.csv', delimiter=';')
        self.mean2 = features.describe().loc[['mean']]
        self.std2 = features.describe().loc[['std']]
        self.features = pd.concat([features, features])
        feature_selection = self.ui.feature_selection.currentIndex()
        if feature_selection == 1:
            self.infor()
            print("Информативность")
        elif feature_selection == 2:
            self.old_correllation()
            print("Корреляция")
        elif feature_selection == 3:
            self.old_correllation()
            self.new_features()
            self.infor()
            print("Корреляция + Информативность")
        elif feature_selection == 4:
            self.infor()
            self.new_features()
            self.old_correllation()
            print("Информативность + Корреляция")

    @timer
    def informativ(self): # Отображение графика информативности
        """
        Вход: режим расчета МПС, цветовой канал, glcm_train0.csv и glcm_train1.csv, расстояние смежности

        :return: график информативности признаков
        """
        np.set_printoptions(edgeitems=100000)
        # Проверка на указание расстояния смежности
        mode = self.ui.modeBox.currentIndex()
        color = self.ui.color.currentIndex()
        if self.spinBox.value()==0:
            print("Укажите расстояние смежности в spinbox")
        features = pd.read_csv('csv/glcm_train0.csv', delimiter=';')
        self.mean1 = features.describe().loc[['mean']]
        self.std1 = features.describe().loc[['std']]
        features = pd.read_csv('csv/glcm_train1.csv', delimiter=';')
        self.mean2 = features.describe().loc[['mean']]
        self.std2 = features.describe().loc[['std']]
        self.rass = self.spinBox.value()  # Расстояние смежности
        c = numpy.array(abs(self.mean1 - self.mean2))
        z = numpy.array(1.6 * (self.std1 + self.std2))
        self.informativeness = numpy.divide(c, z)
        print(len(self.informativeness))
        if mode==0:
            x = 1
            num = 4 * 8
            feature = 8
            ugol = 4
            color_feature = 4 * 8
            markers = ['$A$','$B$','$C$','$D$','$E$','$F$','$G$','$H$']
            if color == 0:
                color = ['red']
            elif color == 1:
                color = ['green']
            elif color == 2:
                color = ['blue']
            elif color == 3:
                color = ['gray']
            elif color == 4:
                x = 4
                num = 4 * 4 * 8
                color = ['red', 'green', 'blue', 'gray']
            self.massive=np.array(range(1,num+1)).flatten()
            print(self.massive)
            self.informativeness = self.informativeness[0].reshape(x,color_feature)
            self.massive = self.massive.reshape(x,color_feature)
            print(self.informativeness)
            for i in range(x):
                self.color = self.informativeness[i].reshape(feature,ugol)
                self.arange = self.massive[i].reshape(feature,ugol)
                print(self.color)
                for g in range(8):
                    plt.grid(axis='both')
                    plt.title("Зависимость информативности от номера текстурного признака")
                    print(self.arange[g],self.color[g])
                    plt.plot(self.arange[g], self.color[g], marker=markers[g], color=color[i])

                    self.Anglelegend = ['Контраст', 'Несходство', 'Локальная однородность', 'Угловой второй момент',
                                        'Энергия',
                                        'Корреляция', 'Энтропия', 'Максимум вероятности']
                    plt.figlegend(self.Anglelegend)
                    plt.show()
        elif mode==1:
            tsvet = 1
            feature = 4
            ras_num_text = 4 * 8
            num = 4 * 8 * self.rass
            markers = ['$A$', '$B$', '$C$', '$D$', '$E$', '$F$', '$G$', '$H$']
            if color == 0:
                color = ['red']
            elif color == 1:
                color = ['green']
            elif color == 2:
                color = ['blue']
            elif color == 3:
                color = ['gray']
            elif color == 4:
                tsvet = 4
                feature = 4
                ras_num_text = 4 * 4 * 8
                num = 4 * 4 * 8 * self.rass
                color = ['red', 'green', 'blue', 'gray']

            self.massive = np.array(range(1, num + 1)).flatten()
            self.rass_smez = self.informativeness[0].reshape(self.rass,ras_num_text)
            self.rass_smez_massive = self.massive.reshape(self.rass,ras_num_text)
            for i in range(self.rass):
                self.colors = self.rass_smez[i].reshape(tsvet,32)
                self.colors_massive = self.rass_smez_massive[i].reshape(tsvet,32)
                for j in range(tsvet):
                    self.features = self.colors[j].reshape(8,4)
                    self.features_massive = self.colors_massive[j].reshape(8, 4)
                    for k in range(feature):
                        plt.plot(self.features_massive[k], self.features[k], marker=markers[k], color=color[j])
                        plt.grid(axis='both')
                        plt.title("Зависимость информативности от расстояния смежности")
                        self.Anglelegend = ['Контраст', 'Несходство', 'Локальная однородность', 'Угловой второй момент',
                                            'Энергия',
                                            'Корреляция', 'Энтропия', 'Максимум вероятности']
                        plt.figlegend(self.Anglelegend)
                        plt.show()
        print("Время выполнения работы функции информативность признаков в общем:")


    def UMCG(self):
        """
        Вход: два .csv файла с текстурными признаками ( glcm_train0.csv и glcm_train1.csv )

        :return: x_train,y_train,x_test,y_test - массив значений (x) обучающей выборки,тестовой выборки и (y) значения классов
        """
        self.max_features = None
        feature_selection = self.ui.feature_selection.currentIndex()
        cell1 = pd.read_csv("csv/glcm_train0.csv", delimiter=';')
        print(feature_selection)
        if feature_selection == 1:
            self.load2class()
            cell1 = cell1.iloc[:, self.max_features]
        cell1["class"] = 0
        cell2 = pd.read_csv("csv/glcm_train1.csv", delimiter=';')
        if  feature_selection == 1:
            self.load2class()
            cell2 = cell2.iloc[:, self.max_features]
        cell2["class"] = 1
        self.all_cells = pd.concat([cell1, cell2])
        print(self.all_cells)
        self.cells = self.all_cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))

        test_cell1 = pd.read_csv("csv/glcm_test0.csv", delimiter=';')
        if feature_selection == 1:
            self.load2class()
            test_cell1 = test_cell1.iloc[:, self.max_features]
        test_cell1["class"] = 0
        test_cell2 = pd.read_csv("csv/glcm_test1.csv", delimiter=';')
        if feature_selection == 1:
            self.load2class()
            test_cell2 = test_cell2.iloc[:, self.max_features]
        test_cell2["class"] = 1
        self.all_test_cells = pd.concat([test_cell1, test_cell2])
        self.test_cells = self.all_test_cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
        # Соединение и преобразование выборок
        self.x_train = self.cells.drop(columns=self.cells.columns[-1]).to_numpy()
        self.y_train = self.all_cells.iloc[:, -1:].to_numpy().flatten()
        self.x_test = self.test_cells.drop(columns=self.test_cells.columns[-1]).to_numpy()
        self.y_test = self.all_test_cells.iloc[:, -1:].to_numpy().flatten()

        self.x_train = preprocessing.normalize(self.x_train)
        self.x_test = preprocessing.normalize(self.x_test)

    @timer
    def MLA(self):
        """
        Вход: x_train,y_train,x_test,y_test

        :return: значения метрик моделей, сохраненные модели .pkl
        """
        my_file = open("csv/metrics.csv", "w+")
        my_file.close()
        file_csv_path = 'csv/metrics.csv'
        os.remove(file_csv_path)
        self.UMCG()
        # Настройка параметров оценивания алгоритма
        seed = 42
        num_folds = 5
        scoring = 'accuracy' # метрика оценивания модели
        ###############################################################################
        # Блиц-проверка алгоритмов машинного обучения (далее - алгоритмов) на исходных,
        # необработанных, данных
        models = []
        models.append(('LR', LogisticRegression())) # Линейные модели
        models.append(('LDA', LDA())) # Линейный дискриминантный анализ
        models.append(('KNN', KNeighborsClassifier())) # Ближайший сосед
        models.append(('CART', DecisionTreeClassifier())) # Деревья решений
        models.append(('NB', GaussianNB())) # Наивный Байес
        models.append(('SVC', SVC())) # Машины Опорных Векторов
        models.append(('MLP', MLPClassifier())) # Многослойный перцептрон
        # Оценивание эффективности выполнения каждого алгоритма
        scores = []
        names = []
        results = []
        predictions = []
        msg_row = []
        metrics = []
        ROC_AUC_ALL = []
        for name, model in models:
            kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
            cv_results = cross_val_score(model, self.x_train, self.y_train, cv=kfold, scoring=
            scoring)
            names.append(name)
            results.append(cv_results)
            m_fit = model.fit(self.x_train, self.y_train)
            with open(f'models/{name}.pkl', 'wb') as fid:
                joblib.dump(m_fit, fid)
            m_predict = model.predict(self.x_test)
            predictions.append(m_predict)
            m_score = model.score(self.x_test, self.y_test)
            scores.append(m_score)
            y_pred = []
            for i in range(len(self.x_test)):
                y_pred.append(m_fit.predict([self.x_test[i]]))
            y_pred = np.concatenate(y_pred)
            Accuracy = accuracy_score(self.y_test, y_pred)
            print(self.y_test)
            print(y_pred)
            precision, recall, fscore, support = score(self.y_test, y_pred)
            ROC_AUC = roc_auc_score(self.y_test, y_pred)
            metrics.append([name,Accuracy,precision[1],recall[1],fscore[1],ROC_AUC])
            print("Точность:", Accuracy)
            print('Прецизионность: {}'.format(precision))
            print('Отзыв: {}'.format(recall))
            print('F-мера: {}'.format(fscore))
            print('Поддержка: {}'.format(support))
            print("ROC_AUC: {}".format(ROC_AUC))
            msg = "%s: train = Средняя %.4f (std %.4f) / test = %.4f" % (name, cv_results.mean(),
                                                             cv_results.std(), m_score)
            msg_row.append(msg)
            print(msg)
            ROC_AUC_ALL.append(ROC_AUC)
        self.ui.ROC_LR.setNum(ROC_AUC_ALL[0])  # отображение точности
        self.ui.ROC_LDA.setNum(ROC_AUC_ALL[1])  # отображение точности
        self.ui.ROC_KNN.setNum(ROC_AUC_ALL[2])  # отображение точности
        self.ui.ROC_CART.setNum(ROC_AUC_ALL[3])  # отображение точности
        self.ui.ROC_NB.setNum(ROC_AUC_ALL[4])  # отображение точности
        self.ui.ROC_SVC.setNum(ROC_AUC_ALL[5])  # отображение точности
        self.ui.ROC_MLP.setNum(ROC_AUC_ALL[6])  # отображение точности
        metrics = pd.DataFrame(metrics,columns=['models','Acc', 'Precision', 'Recall', 'F-score','ROC-AUC'])
        metrics.to_csv('csv/metrics.csv', float_format="%.5f", sep=';')
        # Диаграмм размаха («ящик с усами»)
        fig = plt.figure()
        fig.suptitle('Сравнение результатов выполнения алгоритмов')
        ax = fig.add_subplot(111)
        red_square = dict(markerfacecolor='r', marker='s')
        plt.boxplot(results, flierprops=red_square)
        ax.set_xticklabels(names, rotation=45)
        plt.show()
        print("Время расчета моделей:")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

