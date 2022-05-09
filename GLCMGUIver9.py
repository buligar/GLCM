import time # библиотека времени
import cv2 # библиотека компьютерного зрения
import numpy # расширение языка Python, добавляющее поддержку больших многомерных массивов и матриц, вместе с большой библиотекой высокоуровневых математических функций для операций с этими массивами.
import matplotlib # библиотека для визуализации данных двумерной графикой
import os # предоставляет функции для взаимодействия с операционной системой
import shutil # модуль предлагает ряд высокоуровневых операций с файлами и коллекциями файлов.
import sys # Не удалять!!!
import seaborn as sns #Seaborn — это библиотека визуализации данных Python, основанная на matplotlib .
import np as np # все модули numpy и можете использовать их как np.
import numpy as np # все модули numpy и можете использовать их как np.
import pandas as pd # популярный инструментарий анализа данных
import matplotlib.pylab as plt # набор функций командного стиля, которые заставляют matplotlib работать как MATLAB.
import skimage.segmentation # модуль сегментации scikit-image
import skimage.filters.edges # модуль детектора границ
import sklearn.utils._typedefs # Не удалять !!!
import sklearn.neighbors._partition_nodes # Не удалять !!!
import tensorflow as tf # библиотека Tensorflow для нейросетей
from PIL import Image # Библиотека для работы с изображениями
from scipy import stats # Библиотека для научных и технических вычислений.
from glob import glob # для работы с путями
from PyQt5 import QtCore, QtWidgets, uic # QTCore - Модуль содержит основные классы, в том числе цикл событий и механизм сигналов и слотов Qt. Вспомогательные модули, для работы с виджетами и ui-фалйами, сгенерированными в дизайнере
from PyQt5.QtWidgets import QMenuBar, QMenu, QFileDialog,QGraphicsScene  # подключение виджетов
from PyQt5.QtGui import QPixmap # подключение модуля для работы с изображениями
from skimage.feature import greycomatrix, greycoprops# подключение модуля для построения МПС и вычисления текстурных признаков
from matplotlib import pyplot as plt # Pyplot предоставляет интерфейс конечного автомата для базовой библиотеки построения графиков в matplotlib.
from tkinter import * # пакет для Python, предназначенный для работы с библиотекой Tk
from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
from keras.models import Sequential # Для создания слоев
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D # слои нейросети
from tensorflow.keras.utils import CustomObjectScope # Предоставляет пользовательские классы/функции внутренним компонентам десериализации Keras.
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split # Оценка баллов перекрестной проверкой и разделение на тренировочную и тестовую выборку
from sklearn.feature_selection import f_classif,SelectPercentile,RFECV # Вычисление F-значение ANOVA для предоставленного образца и выбор функции в соответствии с процентилем наивысших оценок.
from sklearn import preprocessing # предоставляет несколько общих служебных функций и классов преобразования для преобразования необработанных векторов признаков в представление, более подходящее для последующих оценок.
from sklearn.preprocessing import StandardScaler,LabelEncoder # Стандартизируйте функции, удалив среднее значение и масштабируя до единичной дисперсии и LabelEncoder можно использовать для нормализации меток.
from sklearn.pipeline import make_pipeline, Pipeline # это служебная функция, которая является сокращением для построения конвейеров.
from sklearn.svm import SVC # SVM классификатор
from sklearn.neural_network import MLPClassifier # MLP классификатор
from sklearn.neighbors import KNeighborsClassifier # KNN классификатор
from sklearn.metrics import confusion_matrix # Вычислите матрицу путаницы, чтобы оценить точность классификации.
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn import metrics
import sklearn.feature_selection as fs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,roc_curve, RocCurveDisplay,f1_score,jaccard_score,precision_score
from metrics import dice_coef, iou # функции потери коэффициента и индексом Жаккара, по сути является методом количественной оценки процентного перекрытия между целевой маской и нашим прогнозируемым результатом.
from train import dataset_path_train_segmentation, create_dir_train_segmentation # подключение функций создания путей и обучения сегментации
from eval import dataset_path_test_segmentation # подключение проверки модели сегментации

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # INFO и WARNING сообщения не печатаются
matplotlib.use('QT5Agg') # подключение бэкенда QTAgg, представляет собой неинтерактивный бэкенд, который может записывать только в файлы

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self): # Инициализация
        super(MyWindow, self).__init__()
        self.ui=uic.loadUi('GLCM.ui', self) # Импорт интерфейса
        self.addFunctions() # Вызов функций

    # Декоратор подсчета времени выполнения функции
    def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            val = func(*args, **kwargs)
            print(f"{time.time()-start}")
            return val
        return wrapper

    # Подключение функций взаимодействия с интерфейсом
    def addFunctions(self):
        # Menu
        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)
        fileMenu = QMenu("&Файл", self)
        self.menuBar.addMenu(fileMenu)
        fileMenu.addAction('Открыть',self.action_clicked)
        fileMenu.addAction('Загрузить обучающую выборку', self.action_clicked)
        fileMenu.addAction('Загрузить тестовую выборку', self.action_clicked)
        fileMenu.addAction('Загрузить метаданные .csv', self.action_clicked)
        fileMenu.addAction('Загрузить всю выборку', self.action_clicked)
        self.listWidget.itemClicked.connect(self.listitemclicked) # load image
        self.pushButton_deletehair.clicked.connect(lambda: self.removehair())
        self.pushButton_segmentation.clicked.connect(lambda: self.segmentation_OTSU())  # Сегментация
        self.pushButton.clicked.connect(lambda: self.glcm_one()) # GLCM
        self.pushButton_2.clicked.connect(lambda: self.glcm_all()) # GLCM обучающая и тестовая выборка
        self.pushButton_deleteBackground.clicked.connect(lambda: self.deletebackground()) # Удаление фона
        self.info_button.clicked.connect(lambda: self.informativ()) # Информативность в общем
        self.info_button_2.clicked.connect(lambda: self.informativ2()) # Информативность по отдельности
        self.zoom_in_button.clicked.connect(lambda: self.on_zoom_in()) # Увеличить
        self.zoom_out_button.clicked.connect(lambda: self.on_zoom_out()) # Уменьшить
        self.zoom_in_button_2.clicked.connect(lambda: self.on_zoom_in2())  # Увеличить
        self.zoom_out_button_2.clicked.connect(lambda: self.on_zoom_out2())  # Уменьшить
        self.MLP_button.clicked.connect(lambda: self.MLP()) # MLP Классификатор
        self.KNN_button.clicked.connect(lambda: self.KNN()) # KNN Классификатор
        self.SVM_button.clicked.connect(lambda: self.SVM())  # SVM Классификатор
        self.NN_button.clicked.connect(lambda: self.NN()) # NN классификатор
        self.filters_button.clicked.connect(lambda: self.filters())
        self.pushButton_segmentation_2.clicked.connect(lambda: self.segmentation_Unet()) # Сегментация U-net
        self.pushButton_train_segmentation_U_net.clicked.connect(lambda: self.train_segmentation_Unet()) # Обучение U-net
        self.pushButton_test_segmentation_U_net.clicked.connect(lambda: self.test_segmentation_Unet()) # Тестирование U-net
        self.pushButton_delete_items.clicked.connect(lambda: self.deleteitems()) # удаление всех элементов из списка
        self.pushButton_delete_item.clicked.connect(lambda: self.deleteitem()) # удаление элемента из списка

    @QtCore.pyqtSlot()
    def action_clicked(self):  # Строка меню
        action = self.sender()
        if action.text() == "Открыть":
            try:
                self.files_path = QFileDialog.getOpenFileNames(self)[0]  # извлечение изображений из Dialog окна
                print(self.files_path)
                self.listWidget.addItems(self.files_path)  # добавление изображений в список
            except FileNotFoundError:
                print("No such file")
        elif action.text() == "Загрузить обучающую выборку":
            try:
                self.train_papka=QFileDialog.getExistingDirectory(self) # Путь к папке содержащей папки классов
                print(self.train_papka)
                self.pred_train_papka=os.path.dirname(self.train_papka) # Путь к папке куда сохранять .csv
                print(self.pred_train_papka)
            except NotADirectoryError:
                print("No such directory")
        elif action.text() == "Загрузить тестовую выборку":
            try:
                self.test_papka=QFileDialog.getExistingDirectory(self) # Путь к папке содержащей папки классов
                print(self.test_papka)
                self.pred_test_papka=os.path.dirname(self.test_papka) # Путь к папке куда сохранять .csv
                print(self.pred_test_papka)
            except NotADirectoryError:
                print("No such directory")
        elif action.text() == "Загрузить метаданные .csv":
            try:
                self.metadata = QFileDialog.getOpenFileName(self,filter = "csv(*.csv)")[0]
                print(self.metadata)
            except NotADirectoryError:
                print("No such directory")
        elif action.text() == "Загрузить всю выборку":
            try:
                self.dataset=QFileDialog.getExistingDirectory(self) # Путь к папке содержащей папки классов
                print(self.dataset)
            except NotADirectoryError:
                print("No such directory")


    def listitemclicked(self): # Выбор из списка изображения и отображение на label

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

    def deleteitem(self):
        listItems = self.listWidget.selectedItems()
        if not listItems: return
        for item in listItems:
            self.listWidget.takeItem(self.listWidget.row(item))
        self.scene.clear()
        self.scene_2.clear()
    def deleteitems(self): # Удаление элемента из списка
        try:
            self.listWidget.clear()
            self.scene.clear()
            self.scene_2.clear()
        except:
            print('Список уже пуст')
    def on_zoom_in(self): # Увеличение изображения 1 окна
        self.scene.clear()
        self.scale *= 2
        self.resize_image()

    def on_zoom_out(self): # Уменьшение изображения 1 окна
        self.scene.clear()
        self.scale /= 2
        self.resize_image()

    def on_zoom_in2(self): # Увеличение изображения 2 окна
        self.scene_2.clear()
        self.scale *= 2
        self.resize_image2()

    def on_zoom_out2(self): # Уменьшение изображения 2 окна
        self.scene_2.clear()
        self.scale /= 2
        self.resize_image2()

    def resize_image(self): # Изменение изображения 1 окна
        size = self.pixmap.size()
        scaled_pixmap = self.pixmap.scaled(self.scale * size)
        self.scene.addPixmap(scaled_pixmap)

    def resize_image2(self): # Изменение изображения 2 окна
        size = self.pixmap2.size()
        scaled_pixmap = self.pixmap2.scaled(self.scale * size)
        self.scene_2.addPixmap(scaled_pixmap)

    def removehair(self):

        image = cv2.imread(self.file_path)
        image_resize = cv2.resize(image, (1800, 1200))
        grayScale = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY) # Преобразование исходного изображения в оттенки серого
        kernel = cv2.getStructuringElement(1, (17, 17)) # Ядро для морфологической фильтрации
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) # Выполните фильтрацию черной шляпы на изображении в градациях серого, чтобы найти контуры волос.
        ret, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY) # усилить контуры волос при подготовке к окрашиванию
        final_image = cv2.inpaint(image_resize, threshold, 1, cv2.INPAINT_TELEA) # закрасить исходное изображение в зависимости от маски
        cv2.imwrite("withouthair.png", final_image)
        self.pixmap2 = QPixmap('withouthair.png')
        self.scene_2.addPixmap(self.pixmap2)
        self.file_path = 'withouthair.png'


    def segmentation_OTSU(self): # Сегментация Оцу
        np.set_printoptions(edgeitems=100000)
        img = cv2.imread(self.file_path)
        print(img)
        gray = cv2.imread(self.file_path,0)

        th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU ) # Порог
        # Найти первый контур больше 100, расположенный в центральной области
        cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea)
        H, W = img.shape[:2]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 100 and (1 < w / h < 2) and (W / 2 < x + w // 2 < W * 1 / 2) and (H / 1 < y + h // 2 < H * 1 / 2):
                break
        # Создать маску и выполнить побитовую операцию
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        cv2.imwrite("masks_OTSU/mask.png", mask)
        dst = cv2.bitwise_and(img, img, mask=mask)


        # Вывод на экран
        cv2.imwrite("dst.png", dst)
        self.scene_2 = QGraphicsScene(self.graphicsView_2)
        self.pixmap2 = QPixmap('dst.png')
        self.scene_2.addPixmap(self.pixmap2)
        self.graphicsView_2.setScene(self.scene_2)
        self.file_path='dst.png'


        ### 2
        # img = cv2.imread(self.file_path, 0)
        # # global thresholding
        # ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # # Otsu's thresholding
        # ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # Otsu's thresholding after Gaussian filtering
        # blur = cv2.GaussianBlur(img, (5, 5), 0)
        # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ### Вариант 1
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        # erosion = cv2.erode(th3, kernel, iterations=1)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        # dilation = cv2.dilate(erosion, kernel, iterations=1)
        ### Вариант 2
        # se1 = cv2.getStructuringElement(cv2.MORPH_ERODE, (5, 5))
        # se2 = cv2.getStructuringElement(cv2.MORPH_DILATE, (2, 2))
        # mask = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, se1)
        # dilation = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

        # images = [img, 0, th1,
        #           img, 0, th2,
        #           blur, 0, dilation]
        # titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
        #           'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
        #           'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
        # for i in range(3):
        #     plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        #     plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        #     plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        #     plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        #     plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        #     plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
        # plt.show()
    def train_segmentation_Unet(self): # Обучение сегментации U-net
        # dataset_path="D:/Downloads/ISIC2018/"
        dataset_path="input/ISIC2018/"
        Height = self.Height.value()  # Не удалять !!!
        Width = self.Width.value()  # Не удалять !!!
        batch_size = self.bs.value()
        lr = float(self.ui.lr.currentText())
        num_epochs = self.num_epochs.value()
        dataset_path_train_segmentation(batch_size,lr,num_epochs,Height,Width,dataset_path)
    def test_segmentation_Unet(self): # Тестирование сегмментации U-net
        # dataset_path = "D:\\Downloads\\ISIC2018\\"
        dataset_path="input\\HAM10000\\"
        Height = self.Height.value()  # Не удалять !!!
        Width = self.Width.value()  # Не удалять !!
        dataset_path_test_segmentation(Height,Width,dataset_path)

    def segmentation_Unet(self): # Сегментация c загруженной моделью нейронной сети U-net
        np.set_printoptions(edgeitems=100000)
        np.random.seed(42)
        tf.random.set_seed(42)
        create_dir_train_segmentation("masks") # Создание папки masks
        try:
            H = self.Height.value()  # Не удалять !!!
            W = self.Width.value()  # Не удалять !!!
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):  # Загрузка модели
                model = tf.keras.models.load_model("files/model.h5")

            # Загрузка изображения
            img = cv2.imread(self.file_path, cv2.IMREAD_COLOR)  # Загрузка изображения
            img = cv2.resize(img, (W, H))  # Изменить размер
            image_resize = np.array(img, dtype=np.uint8)  # Преобразовать в массив
            x = img / 255.0  # Нормализация [0,1]
            x = x.astype(np.float32)  # Преобразовать во float
            x = np.expand_dims(x, axis=0)

            # Прогнозирование маски
            y_pred = model.predict(x)[0] > 0.5  # Предугадать по модели
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred.astype(np.uint8)
            dst = cv2.bitwise_and(image_resize, image_resize, mask=y_pred)

            # Вывод на экран
            cv2.imwrite("dst.png", dst)
            self.scene_2 = QGraphicsScene(self.graphicsView_2)
            self.pixmap2 = QPixmap('dst.png')
            self.scene_2.addPixmap(self.pixmap2)
            self.graphicsView_2.setScene(self.scene_2)
            self.file_path = 'dst.png'
        except:
            print("Укажите размеры изображения")



    @timer
    def deletebackground(self): # Удаления фона

        img = cv2.imread(self.file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # преобразовать в серый цвет
        # порог
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        # применяем морфологию для очистки небольших пятен
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        # получаем внешний контур
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        # рисуем белый залитый контур на черном фоне как мас
        contour = np.zeros_like(gray)
        cv2.drawContours(contour, [big_contour], 0, 255, -1)
        # размытие увеличить изображение
        blur = cv2.GaussianBlur(contour, (5, 5), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
        # растянуть так, чтобы 255 -> 255 и 127,5 -> 0
        mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5, 255), out_range=(0, 255))
        # поместить маску в альфа-канал ввода
        self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        self.result[:, :, 3] = mask
        # сохранить вывод
        cv2.imwrite('withoutBackground.png', self.result)
        # Вывод на экран
        self.scene_2.clear()
        self.pixmap2 = QPixmap('withoutBackground.png')
        self.scene_2.addPixmap(self.pixmap2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Время выполнения работы функции удаление фона:")

    def filters(self):
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5, 5), 0)

    def glcm_vn(self,grayscale):

        if self.tsvet == 0:
            grayscale[:, :, 0] = 0
            grayscale[:, :, 1] = 0
            cv2.imwrite('r.bmp', grayscale)
            # self.scene_2.clear()
            self.pixmap2 = QPixmap('r.bmp')
            self.scene_2.addPixmap(self.pixmap2)
            self.graphicsView_2.setScene(self.scene_2)
        elif self.tsvet == 1:
            grayscale[:, :, 0] = 0
            grayscale[:, :, 2] = 0
            cv2.imwrite('g.bmp', grayscale)
            # self.scene_2.clear()
            self.pixmap2 = QPixmap('g.bmp')
            self.scene_2.addPixmap(self.pixmap2)
            self.graphicsView_2.setScene(self.scene_2)
            # self.graphicsView_2.
        elif self.tsvet == 2:
            grayscale[:, :, 1] = 0
            grayscale[:, :, 2] = 0
            cv2.imwrite('b.bmp', grayscale)
            # self.scene_2.clear()
            self.pixmap2 = QPixmap('b.bmp')
            self.scene_2.addPixmap(self.pixmap2)
            self.graphicsView_2.setScene(self.scene_2)
        elif self.tsvet == 3:
            grayscale = cv2.imread(self.file_path, 0)
            cv2.imwrite('gray.bmp', grayscale)
            # self.scene_2.clear()
            self.pixmap2 = QPixmap('gray.bmp')
            self.scene_2.addPixmap(self.pixmap2)
            self.graphicsView_2.setScene(self.scene_2)

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
        print(self.Contrast)

        # из двумерного массива в одномерный
        Contrast = np.concatenate(self.Contrast)
        Dissimilarity = np.concatenate(self.Dissimilarity)
        Homogeneity = np.concatenate(self.Homogeneity)
        Asm = np.concatenate(self.Asm)
        Energy = np.concatenate(self.Energy)
        Correlation = np.concatenate(self.Correlation)
        Entropy = np.concatenate(self.Entropy)
        print(Contrast)

        # Сохранение по отдельности
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

        # Сохранение все вместе
        self.GLCM_All = [Contrast] + [Dissimilarity] + [Homogeneity] + [Asm] + [Energy] + [Correlation] + [Entropy]
        self.GLCM_All = np.concatenate(self.GLCM_All)
        self.GLCM_RGB.append(self.GLCM_All)
        print(self.GLCM_RGB)


    @timer
    def glcm_one(self):

        # Очищение файла glcm_all.csv
        my_file = open("glcm_all.csv", "w+")
        my_file.close()
        file_csv_path = 'glcm_all.csv'
        os.remove(file_csv_path)
        # # Очищение папки texture
        # path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'texture')
        # shutil.rmtree(path)
        # os.mkdir("texture")
        mode = self.ui.modeBox.currentIndex()
        color = self.ui.color.currentIndex()
        if mode == 0:
            if color == 4:
                self.features_sum = 7 * 4 * 4
                head = np.array(range(1, self.features_sum + 1)).flatten()
                self.Distances = [self.spinBox.value()]
                value = self.spinBox.value()
                Distances = np.arange(1, value + 1, 1)
            else:
                self.features_sum = 7 * 4
                head = np.array(range(1, self.features_sum + 1)).flatten()
                self.Distances = [self.spinBox.value()]
                value = self.spinBox.value()
                Distances = np.arange(1, value + 1, 1)
        elif mode == 1:
            if color == 4:
                self.features_sum = 7 * 4 * 4 * self.spinBox.value()
                head = np.array(range(1, self.features_sum + 1)).flatten()
                value = self.spinBox.value()
                self.Distances = np.arange(1, value + 1, 1)
                Distances = np.arange(1, value + 1, 1)
            else:
                self.features_sum = 7 * 4 * self.spinBox.value()
                head = np.array(range(1, self.features_sum + 1)).flatten()
                value = self.spinBox.value()
                self.Distances = np.arange(1, value + 1, 1)
                Distances = np.arange(1, value + 1, 1)
        num = len(self.files_path) # Всего изображений
        self.num_cells=np.array(range(1,num+1)).flatten() # Массив кол-ва изображений
        df = pd.DataFrame(np.matrix(head)) # Создание заголовков столбцов
        df.to_csv('glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';') # Указание кол-ва признаков
        select_segmentation = self.ui.select_segmentation.currentIndex()
        for i in range(num):
            self.file_path=self.files_path[i]
            print(self.file_path)
            self.segmentation_Unet()
            if select_segmentation == 0:
                np.set_printoptions(edgeitems=1000) # для отображения полной матрицы изображения в окне вывода
                self.file_path = 'dst.png'
                self.GLCM_RGB = []
                # Сохранение и отображение канала цвета
                if color < 4:
                    self.tsvet = color
                    grayscale = cv2.imread(self.file_path)  # Преобразование в оттенки серого
                    self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
                elif color == 4 :
                    for k in range(4):
                        self.tsvet = k
                        grayscale = cv2.imread(self.file_path)
                        self.glcm_vn(grayscale)

                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
            else:
                np.set_printoptions(edgeitems=1000)  # для отображения полной матрицы изображения в окне вывода
                self.file_path = 'dst.png'
                self.GLCM_RGB = []
                # Сохранение и отображение канала цвета
                if color < 4:
                    self.tsvet = color
                    grayscale = cv2.imread(self.file_path)  # Преобразование в оттенки серого
                    self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
                elif color == 4:
                    for k in range(4):
                        self.tsvet = k
                        grayscale = cv2.imread(self.file_path)
                        self.glcm_vn(grayscale)
                    self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
                    mat = np.matrix(self.GLCM_RGB)
                    df = pd.DataFrame(mat)
                    df.to_csv('glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
        #Отображение на графиках

        plt.clf()
        plt.subplot(2, 4, 1)
        plt.grid(axis='both')
        plt.title("Контраст")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(self.Distances, self.Contrast, marker='o')

        plt.subplot(2, 4, 2)
        plt.grid(axis='both')
        plt.title("Несходство")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(self.Distances, self.Dissimilarity, marker='o')

        plt.subplot(2, 4, 3)
        plt.grid(axis='both')
        plt.title("Локальная однородность")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(self.Distances, self.Homogeneity, marker='o')

        plt.subplot(2, 4, 4)
        plt.grid(axis='both')
        plt.title("Угловой второй момент")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(self.Distances, self.Asm, marker='o')

        plt.subplot(2, 4, 5)
        plt.grid()
        plt.title("Энергия")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(self.Distances, self.Energy, marker='o')

        plt.subplot(2, 4, 6)
        plt.grid(axis='both')
        plt.title("Корреляция")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(self.Distances, self.Correlation, marker='o')

        plt.subplot(2, 4, 7)
        plt.grid(axis='both')
        plt.title("Энтропия")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(self.Distances, self.Entropy, marker='o')

        self.Anglelegend = [0, 45, 90, 135]

        plt.figlegend(self.Anglelegend)
        plt.show()
        self.features_describe()
        print("Время выполнения работы функции расчета признаков с учетом параметров:")

    @timer
    def glcm_all(self):
        path = self.train_papka
        list = os.listdir(self.train_papka)
        print(list)
        for k in range(10):
            my_file = open(f"glcm_train{k}.csv", "w+")
            my_file.close()
            file_csv_path = f'glcm_train{k}.csv'
            os.remove(file_csv_path)
        file_count = len(list)
        my_file = open("glcm_all.csv", "w+")
        my_file.close()
        file_csv_path = 'glcm_all.csv'
        os.remove(file_csv_path)
        print(file_count)
        for k in range(file_count):
            papka = []
            for p in os.listdir(path):
                full_path = os.path.join(path, p).replace('/','\\')
                papka += [full_path]
            print(papka)
            self.files_path = []
            for l in os.listdir(papka[k]):
                full_path = os.path.join(papka[k], l)
                self.files_path += [full_path]
            print(self.files_path)
            # x_train, x_test = train_test_split(self.files_path, test_size=0.25, random_state=42)
            # print(len(x_train))
            self.glcm_one()
            file_oldname = os.path.join(self.pred_train_papka, "glcm_all.csv")
            file_newname_newfile = os.path.join(self.pred_train_papka,f"glcm_train{k}.csv")
            os.rename(file_oldname, file_newname_newfile)

        path = self.test_papka
        list = os.listdir(self.test_papka)
        folder = os.path.realpath(path)
        print(folder)
        for k in range(10):
            my_file = open(f"glcm_test{k}.csv", "w+")
            my_file.close()
            file_csv_path = f'glcm_test{k}.csv'
            os.remove(file_csv_path)
        file_count = len(list)
        my_file = open("glcm_all.csv", "w+")
        my_file.close()
        file_csv_path = 'glcm_all.csv'
        os.remove(file_csv_path)
        print(file_count)
        for k in range(file_count):
            papka = []
            print(path)
            for p in os.listdir(path):
                full_path = os.path.join(path, p).replace('/','\\')
                papka += [full_path]
            print(papka)
            self.files_path = []
            for l in os.listdir(papka[k]):
                full_path = os.path.join(papka[k], l)
                self.files_path += [full_path]
            self.glcm_one()
            file_oldname = os.path.join(self.pred_test_papka, "glcm_all.csv")
            file_newname_newfile = os.path.join(self.pred_test_papka, f"glcm_test{k}.csv")
            os.rename(file_oldname, file_newname_newfile)
        print("Время выполнения работы расчета признаков обучающей и тестовой выборки:")


    def features_describe(self): # Сохранение описания значений таблицы (mean,std)
        features = pd.read_csv('glcm_all.csv', delimiter=';')
        describe = features.describe()
        df = pd.DataFrame(describe)
        df.to_csv('glcm_describe.csv')

    @timer
    def informativ(self): # вычисление и отображение среднего значения, среднеквадратического значения и информативности в общем
        np.set_printoptions(edgeitems=100000)
        # Проверка на указание расстояния смежности
        mode = self.ui.modeBox.currentIndex()
        if self.spinBox.value()==0:
            print("Укажите расстояние смежности в spinbox")
        elif mode==0:
            self.features_sum = 7*4 * self.spinBox.value()
            self.rass = self.spinBox.value()  # Расстояние смежности
            self.rassipriznaki = 7 * self.rass  # Расстояние смежности * 7 признаков
            x = 4 * 4
        elif mode==1:
            self.features_sum = 7 * 4 *4 * self.spinBox.value()
            self.rass = self.spinBox.value()  # Расстояние смежности
            self.rassipriznaki = 7*4 * self.rass  # Расстояние смежности * 7 признаков
            x = 4 * 4 * self.spinBox.value()
        # Загрузка данных
        features = pd.read_csv('glcm_train0.csv',delimiter=';')
        mean1=features.describe().loc[['mean']]
        std1=features.describe().loc[['std']]
        features = pd.read_csv('glcm_train1.csv', delimiter=';')
        mean2 = features.describe().loc[['mean']]
        std2 = features.describe().loc[['std']]
        # Расчет информативности
        c=numpy.array(abs(mean1-mean2))
        z=numpy.array(1.6*(std1+std2))
        self.informativeness=numpy.divide(c,z)
        ar1=np.arange(0,x)
        ar2=np.arange(x,2*x)
        ar3=np.arange(2*x,3*x)
        ar4=np.arange(3*x,4*x)
        ar5=np.arange(4*x,5*x)
        ar6=np.arange(5*x,6*x)
        ar7 = np.arange(6 * x, 7 * x)
        df1 = self.informativeness[0][ar1]
        df2 = self.informativeness[0][ar2]
        df3 = self.informativeness[0][ar3]
        df4 = self.informativeness[0][ar4]
        df5 = self.informativeness[0][ar5]
        df6 = self.informativeness[0][ar6]
        df7 = self.informativeness[0][ar7]
        arange1 = np.arange(1, x+1)
        arange2 = np.arange(x+1, 2 * x+1)
        arange3 = np.arange(2 * x+1, 3 * x+1)
        arange4 = np.arange(3 * x+1, 4 * x+1)
        arange5 = np.arange(4 * x+1, 5 * x+1)
        arange6 = np.arange(5 * x+1, 6 * x+1)
        arange7 = np.arange(6 * x+1, 7 * x+1)

        plt.grid(axis='both')
        plt.title("Зависимость информативности от расстояния смежности")
        plt.plot(arange1, df1,marker='o',color='red')
        plt.plot(arange2, df2,marker='o',color='green')
        plt.plot(arange3, df3, marker='o', color='blue')
        plt.plot(arange4, df4, marker='o', color='yellow')
        plt.plot(arange5, df5, marker='o', color='orange')
        plt.plot(arange6, df6, marker='o', color='purple')
        plt.plot(arange7, df7, marker='o', color='pink')

        self.Anglelegend = ['Контраст','Несходство','Локальная однородность','Угловой второй момент','Энергия','Корреляция','Энтропия']
        plt.figlegend(self.Anglelegend)
        plt.show()
        print("Время выполнения работы функции информативность признаков в общем:")

    @timer
    def informativ2(self):  # Вычисление и отображение информативности признаков по отдельности
        np.set_printoptions(edgeitems=100000)
        # Проверка на указание расстояния смежности
        if self.spinBox.value() == 0:
            print("Укажите расстояние смежности в spinbox")
        self.features_sum = 7*4 * self.spinBox.value()  # сколько всего признаков
        self.rass = self.spinBox.value()  # Расстояние смежности
        self.rassinapravlenie = 4 * self.rass  # Расстояние смежности * 4 направления
        # Загрузка данных
        features = pd.read_csv('glcm_train0.csv', delimiter=';')
        del features['0']
        mean1 = features.describe().loc[['mean']]
        std1 = features.describe().loc[['std']]
        features = pd.read_csv('glcm_train1.csv', delimiter=';')
        del features['0']
        mean2 = features.describe().loc[['mean']]
        std2 = features.describe().loc[['std']]
        # Расчет информативности
        c = numpy.array(abs(mean1 - mean2))
        z = numpy.array(1.6 * (std1 + std2))
        self.informativeness = numpy.divide(c, z)
        print(self.informativeness)
        infoContrast = np.transpose(self.informativeness.reshape(7, self.rassinapravlenie)[0].reshape(self.rass, 4))
        maxContrast = max(map(max, infoContrast))
        infoDissimilation = np.transpose(
            self.informativeness.reshape(7, self.rassinapravlenie)[1].reshape(self.rass, 4))
        maxDissimilation = max(map(max, infoDissimilation))
        infoHomogeneity = np.transpose(self.informativeness.reshape(7, self.rassinapravlenie)[2].reshape(self.rass, 4))
        maxHomogeneity = max(map(max, infoHomogeneity))
        infoAsm = np.transpose(self.informativeness.reshape(7, self.rassinapravlenie)[3].reshape(self.rass, 4))
        maxAsm = max(map(max, infoAsm))
        infoEnergy = np.transpose(self.informativeness.reshape(7, self.rassinapravlenie)[4].reshape(self.rass, 4))
        maxEnergy = max(map(max, infoEnergy))
        infoCorrelation = np.transpose(self.informativeness.reshape(7, self.rassinapravlenie)[5].reshape(self.rass, 4))
        maxCorrelation = max(map(max, infoCorrelation))
        infoEntropy = np.transpose(self.informativeness.reshape(7, self.rassinapravlenie)[6].reshape(self.rass, 4))
        maxCorrelation = max(map(max, infoCorrelation))
        rass_zmez = np.arange(1, self.spinBox.value() + 1, 1)

        # Отоброжение на графиках
        plt.subplot(2, 3, 1)
        plt.grid(axis='both')
        plt.title("Зависимость информативности контраста\n от расстояния смежности")
        plt.plot(rass_zmez, infoContrast[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 1)
        plt.grid(axis='both')
        plt.title("Зависимость информативности контраста\n от расстояния смежности")
        plt.plot(rass_zmez, infoContrast[1], marker='o')
        plt.show()
        plt.subplot(2, 3, 1)
        plt.grid(axis='both')
        plt.title("Зависимость информативности контраста\n от расстояния смежности")
        plt.plot(rass_zmez, infoContrast[2], marker='o')
        plt.show()
        plt.subplot(2, 3, 1)
        plt.grid(axis='both')
        plt.title("Зависимость информативности контраста\n от расстояния смежности")
        plt.plot(rass_zmez, infoContrast[3], marker='o')
        plt.show()
        plt.subplot(2, 3, 2)
        plt.grid(axis='both')
        plt.title("Зависимость информативности несходства от расстояния смежности")
        plt.plot(rass_zmez, infoDissimilation[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 2)
        plt.grid(axis='both')
        plt.title("Зависимость информативности несходства от расстояния смежности")
        plt.plot(rass_zmez, infoDissimilation[1], marker='o')
        plt.show()
        plt.subplot(2, 3, 2)
        plt.grid(axis='both')
        plt.title("Зависимость информативности несходства от расстояния смежности")
        plt.plot(rass_zmez, infoDissimilation[2], marker='o')
        plt.show()
        plt.subplot(2, 3, 2)
        plt.grid(axis='both')
        plt.title("Зависимость информативности несходства от расстояния смежности")
        plt.plot(rass_zmez, infoDissimilation[3], marker='o')
        plt.show()
        plt.subplot(2, 3, 3)
        plt.grid(axis='both')
        plt.title("Зависимость информативности локальной однородности\n от расстояния смежности")
        plt.plot(rass_zmez, infoHomogeneity[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 3)
        plt.grid(axis='both')
        plt.title("Зависимость информативности локальной однородности\n от расстояния смежности")
        plt.plot(rass_zmez, infoHomogeneity[1], marker='o')
        plt.show()
        plt.subplot(2, 3, 3)
        plt.grid(axis='both')
        plt.title("Зависимость информативности локальной однородности\n от расстояния смежности")
        plt.plot(rass_zmez, infoHomogeneity[2], marker='o')
        plt.show()
        plt.subplot(2, 3, 3)
        plt.grid(axis='both')
        plt.title("Зависимость информативности локальной однородности\n от расстояния смежности")
        plt.plot(rass_zmez, infoHomogeneity[3], marker='o')
        plt.show()
        plt.subplot(2, 3, 4)
        plt.grid(axis='both')
        plt.title("Зависимость информативности углового втрого момента\n от расстояния смежности")
        plt.plot(rass_zmez, infoAsm[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 4)
        plt.grid(axis='both')
        plt.title("Зависимость информативности углового втрого момента\n от расстояния смежности")
        plt.plot(rass_zmez, infoAsm[1], marker='o')
        plt.show()
        plt.subplot(2, 3, 4)
        plt.grid(axis='both')
        plt.title("Зависимость информативности углового втрого момента\n от расстояния смежности")
        plt.plot(rass_zmez, infoAsm[2], marker='o')
        plt.show()
        plt.subplot(2, 3, 4)
        plt.grid(axis='both')
        plt.title("Зависимость информативности углового втрого момента\n от расстояния смежности")
        plt.plot(rass_zmez, infoAsm[3], marker='o')
        plt.show()
        plt.subplot(2, 3, 5)
        plt.grid(axis='both')
        plt.title("Зависимость информативности энергии\n от расстояния смежности")
        plt.plot(rass_zmez, infoEnergy[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 5)
        plt.grid(axis='both')
        plt.title("Зависимость информативности энергии\n от расстояния смежности")
        plt.plot(rass_zmez, infoEnergy[1], marker='o')
        plt.show()
        plt.subplot(2, 3, 5)
        plt.grid(axis='both')
        plt.title("Зависимость информативности энергии\n от расстояния смежности")
        plt.plot(rass_zmez, infoEnergy[2], marker='o')
        plt.show()
        plt.subplot(2, 3, 5)
        plt.grid(axis='both')
        plt.title("Зависимость информативности энергии\n от расстояния смежности")
        plt.plot(rass_zmez, infoEnergy[3], marker='o')
        plt.show()
        plt.subplot(2, 3, 6)
        plt.grid(axis='both')
        plt.title("Зависимость информативности корреляции от расстояния смежности")
        plt.plot(rass_zmez, infoCorrelation[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 6)
        plt.grid(axis='both')
        plt.title("Зависимость информативности корреляции от расстояния смежности")
        plt.plot(rass_zmez, infoCorrelation[1], marker='o')
        plt.show()
        plt.subplot(2, 3, 6)
        plt.grid(axis='both')
        plt.title("Зависимость информативности корреляции от расстояния смежности")
        plt.plot(rass_zmez, infoCorrelation[2], marker='o')
        plt.show()
        plt.subplot(2, 3, 6)
        plt.grid(axis='both')
        plt.title("Зависимость информативности корреляции от расстояния смежности")
        plt.plot(rass_zmez, infoCorrelation[3], marker='o')
        plt.show()
        plt.subplot(2, 3, 7)
        plt.grid(axis='both')
        plt.title("Зависимость информативности энтропии от расстояния смежности")
        plt.plot(rass_zmez, infoEntropy[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 7)
        plt.grid(axis='both')
        plt.title("Зависимость информативности энтропии от расстояния смежности")
        plt.plot(rass_zmez, infoEntropy[1], marker='o')
        plt.show()
        plt.subplot(2, 3, 7)
        plt.grid(axis='both')
        plt.title("Зависимость информативности энтропии от расстояния смежности")
        plt.plot(rass_zmez, infoEntropy[2], marker='o')
        plt.show()
        plt.subplot(2, 3, 7)
        plt.grid(axis='both')
        plt.title("Зависимость информативности энтропии от расстояния смежности")
        plt.plot(rass_zmez, infoEntropy[3], marker='o')
        plt.show()
        self.Anglelegend = [0, 45, 90, 135]

        plt.figlegend(self.Anglelegend)
        plt.show()
        print("Время выполнения работы функции информативность признаков по отдельности:")

    def Loading_samples(self):
        np.set_printoptions(edgeitems=100000)
        # Загрузка обучающей выборки
        cell1 = pd.read_csv("glcm_train0.csv", delimiter=';')
        cell1["class"] = 0
        cell2 = pd.read_csv("glcm_train1.csv", delimiter=';')
        cell2["class"] = 1
        cells = pd.concat([cell1, cell2])
        cells = cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
        print(cells)

        # Загрузка тестовой выборки
        test_cell1 = pd.read_csv("glcm_test0.csv", delimiter=';')
        test_cell1["class"] = 0
        test_cell2 = pd.read_csv("glcm_test1.csv", delimiter=';')
        test_cell2["class"] = 1
        test_cells = pd.concat([test_cell1, test_cell2])
        test_cells = test_cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))

        # Соединение и преобразование выборок
        self.x_train = cells.drop(columns=cells.columns[-1]).to_numpy()
        self.y_train = cells.iloc[:, -1:].to_numpy().flatten()
        self.x_test = test_cells.drop(columns=test_cells.columns[-1]).to_numpy()
        self.y_test = test_cells.iloc[:, -1:].to_numpy().flatten()

        # Нормализация данных
        self.x_train = preprocessing.normalize(self.x_train)
        self.x_test = preprocessing.normalize(self.x_test)

    def my_score(self):
        return f_classif(self.x_train, self.y_train)

    def PCA(self):
        #Уменьшение размерности признаков
        pca = PCA(n_components=10)
        pca.fit(self.x_train)
        self.x_train = pca.transform(self.x_train)
        self.x_test = pca.transform(self.x_test)

    def Gradient(self):
        gb = GradientBoostingClassifier(n_estimators=20)
        gb.fit(self.x_train, self.y_train)
        model = fs.SelectFromModel(gb, prefit=True)
        self.x_train = model.transform(self.x_train)
        self.x_test = model.transform(self.x_test)
        print("The shape of transformed data is {}".format(self.x_train.shape))

    def RFE(self):

        # Create the RFE object and compute a cross-validated score.
        svc = SVC(kernel="linear",cache_size=2000)
        # The "accuracy" scoring shows the proportion of correct classifications

        min_features_to_select = 1  # Minimum number of features to consider
        rfecv = RFECV(
            estimator=svc,
            step=1,
            cv=StratifiedKFold(5),
            scoring="accuracy",
            min_features_to_select=min_features_to_select,
        )
        rfecv.fit(self.x_train, self.y_train)
        self.x_train=rfecv.transform(self.x_train)
        self.x_test=rfecv.transform(self.x_test)
        print(self.x_train)

        print("Оптимальное количество функций : %d" % rfecv.n_features_)

        # # Plot number of features VS. cross-validation scores
        # plt.figure()
        # plt.xlabel("Количество выбранных функций")
        # plt.ylabel("Оценка перекрестной проверки (точность)")
        # plt.plot(
        #     range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
        #     rfecv.grid_scores_,
        # )
        # plt.show()
    @timer
    def SVM(self): # Классификатор опорных векторов

        if self.spinBox_2.value()==0:
            self.ui.output.setText('Укажите номер изображения')
        else:
            self.Loading_samples()
            # Добавьте неинформативные функции
            np.random.seed(0)
            kol_shumov = round((len(self.y_train) * 10) / 100)
            X = np.hstack((self.x_train, 2 * np.random.random((self.x_train.shape[0], kol_shumov))))
            # Создайте преобразование выбора признаков, масштабатор и экземпляр SVM, которые мы объединим вместе, чтобы получить полноценную оценку.
            self.clf = Pipeline(
                [
                    ("anova", SelectPercentile(f_classif)),
                    ("scaler", StandardScaler()),
                    ("svc", SVC(C=1, kernel='rbf',probability=True, cache_size=2000, gamma="auto")),
                ]
            )
            # Постройте оценку перекрестной проверки как функцию процентиля функций
            score_means = list()
            score_stds = list()
            percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
            for percentile in percentiles:
                self.clf.set_params(anova__percentile=percentile)
                this_scores = cross_val_score(self.clf, X, self.y_train)
                score_means.append(this_scores.mean())
                score_stds.append(this_scores.std())
            plt.errorbar(percentiles, score_means, np.array(score_stds))
            plt.title("Производительность SVM-Anova при изменении процентиля выбранных функций")
            plt.xticks(np.linspace(0, 100, 11, endpoint=True))
            plt.xlabel("Процентиль")
            plt.ylabel("Оценка точности")
            plt.axis("tight")
            plt.show()
            # self.RFE()
            self.Gradient()
            # self.PCA()

            # Обучение
            self.clf.fit(self.x_train, self.y_train)
            results=self.clf.score(self.x_test, self.y_test) # измеряет точность модели
            y_pred = []
            for i in range(len(self.x_test)):
                y_pred.append(self.clf.predict([self.x_test[i]]))
            y_pred = np.concatenate(y_pred)
            y_score = self.clf.predict_proba(self.x_test)[:, 1]
            Accuracy = accuracy_score(self.y_test,y_pred)
            Recall = recall_score(self.y_test, y_pred)
            Precision = precision_score(self.y_test, y_pred)
            F1_score = f1_score(self.y_test,y_pred)
            Jaccard = jaccard_score(self.y_test,y_pred)
            ROC_AUC=roc_auc_score(self.y_test, y_score)
            print("Точность:",Accuracy) # Точность измерений — это то, насколько данный набор измерений (наблюдений или показаний) близок или далек от их истинного значения
            print("Отзыв:",Recall)
            print("Прецизионность:",Precision)  # Точность результата измерений — это то, насколько близки или разбросаны измерения друг к другу.
            print("F-мера:",F1_score)
            print("Коэф.Жаккара:", Jaccard)
            print("ROC_AUC:",ROC_AUC)
            fpr, tpr, _ = roc_curve(self.y_test, y_score, pos_label=self.clf.classes_[1])
            RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            plt.show()
            self.ui.accuracy.setNum(results)  # отображение точности
            self.num_test = self.spinBox_2.value()-1 # Номер исследуемого изображения
            res = self.clf.predict([self.x_test[self.num_test]])  # предсказание класса для выбранного изображения из тестовой выборки
            self.ui.results.setNum(res[0] + 1)  # отображение номера класса
            self.show()  # показать на интерфейсе все значения
            print("Время выполнения работы функции SVM:")

    @timer
    def MLP(self):

        self.Loading_samples() # Загрузка данных
        self.clf = MLPClassifier(random_state=1, max_iter=1000).fit(self.x_train, self.y_train)
        results=self.clf.score(self.x_test, self.y_test)
        y_pred = []
        for i in range(len(self.x_test)):
            y_pred.append(self.clf.predict([self.x_test[i]]))
        y_pred = np.concatenate(y_pred)
        y_score = self.clf.predict_proba(self.x_test)[:, 1]
        Accuracy = accuracy_score(self.y_test, y_pred)
        Recall = recall_score(self.y_test, y_pred)
        Precision = precision_score(self.y_test, y_pred)
        F1_score = f1_score(self.y_test, y_pred)
        Jaccard = jaccard_score(self.y_test, y_pred)
        ROC_AUC = roc_auc_score(self.y_test, y_score)
        print("Точность:",
              Accuracy)  # Точность измерений — это то, насколько данный набор измерений (наблюдений или показаний) близок или далек от их истинного значения
        print("Отзыв:", Recall)
        print("Прецизионность:",
              Precision)  # Точность результата измерений — это то, насколько близки или разбросаны измерения друг к другу.
        print("F-мера:", F1_score)
        print("Коэф.Жаккара:", Jaccard)
        print("ROC_AUC:", ROC_AUC)
        fpr, tpr, _ = roc_curve(self.y_test, y_score, pos_label=self.clf.classes_[1])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.show()
        self.ui.accuracy.setNum(results)  # отображение точности
        self.num_test = self.spinBox_2.value() - 1  # Номер исследуемого изображения
        res = self.clf.predict([self.x_test[self.num_test]])  # предсказание класса для выбранного изображения из тестовой выборки
        self.ui.results.setNum(res[0] + 1)  # отображение номера класса
        self.show()  # показать на интерфейсе все значения
        print("Время выполнения работы функции MLP:")

    @timer
    def KNN(self): # Классификатор k-ближайших соседей

        self.Loading_samples() # Загрузка данных
        results = {}
        for i in range(100):
            self.clf = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=i + 2)
            )
            self.clf.fit(self.x_train, self.y_train)
            results[i] = self.clf.score(self.x_test, self.y_test)
        acc = 0.001
        n_neighbors = 0
        for k, v in results.items():
            if v > acc:
                acc = v
                n_neighbors = k
        print("Средняя точность по тестовой выборке:", acc)
        print("Оптимальное количество соседей:", n_neighbors)

        y_pred = []
        for i in range(len(self.x_test)):
            y_pred.append(self.clf.predict([self.x_test[i]]))
        y_pred = np.concatenate(y_pred)
        y_score = self.clf.predict_proba(self.x_test)[:, 1]
        Accuracy = accuracy_score(self.y_test, y_pred)
        Recall = recall_score(self.y_test, y_pred)
        Precision = precision_score(self.y_test, y_pred)
        F1_score = f1_score(self.y_test, y_pred)
        Jaccard = jaccard_score(self.y_test, y_pred)
        ROC_AUC = roc_auc_score(self.y_test, y_score)
        print("Точность:",
              Accuracy)  # Точность измерений — это то, насколько данный набор измерений (наблюдений или показаний) близок или далек от их истинного значения
        print("Отзыв:", Recall)
        print("Прецизионность:",
              Precision)  # Точность результата измерений — это то, насколько близки или разбросаны измерения друг к другу.
        print("F-мера:", F1_score)
        print("Коэф.Жаккара:", Jaccard)
        print("ROC_AUC:", ROC_AUC)
        fpr, tpr, _ = roc_curve(self.y_test, y_score, pos_label=self.clf.classes_[1])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.show()
        self.ui.accuracy.setNum(acc)  # отображение точности
        self.num_test = self.spinBox_2.value() - 1  # Номер исследуемого изображения
        res = self.clf.predict([self.x_test[self.num_test]])  # предсказание класса для выбранного изображения из тестовой выборки
        self.ui.results.setNum(res[0] + 1)  # отображение номера класса
        self.show()  # показать на интерфейсе все значения
        print("Время выполнения работы функции KNN:")

    def NN(self):

        np.random.seed(42)
        skin_df = pd.read_csv(self.metadata)
        SIZE = self.image_size.value()
        print(SIZE)
        batch_size = self.ui.batch_size.currentIndex()
        if batch_size == 0:
            batch_size = 16
        else:
            batch_size = 32
        print(batch_size)
        epochs = self.ui.epochs.value()
        print(epochs)
        n_samples = self.image_norm.value()
        print(n_samples)

        # кодирование метки в числовые значения из текста
        le = LabelEncoder()
        le.fit(skin_df['dx'])
        LabelEncoder()
        # print(list(le.classes_))

        skin_df['label'] = le.transform(skin_df["dx"])
        # print(skin_df.sample(10))

        # Визуализация распределения данных
        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(221)
        skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_ylabel('Count')
        ax1.set_title('Cell Type');

        ax2 = fig.add_subplot(222)
        skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Count', size=15)
        ax2.set_title('Sex');

        ax3 = fig.add_subplot(223)
        skin_df['localization'].value_counts().plot(kind='bar')
        ax3.set_ylabel('Count', size=12)
        ax3.set_title('Localization')

        ax4 = fig.add_subplot(224)
        sample_age = skin_df[pd.notnull(skin_df['age'])]
        sns.distplot(sample_age['age'], fit=stats.norm, color='red');
        ax4.set_title('Age')

        plt.tight_layout()
        plt.show()

        # Распределение данных по различным классам
        from sklearn.utils import resample

        # print(skin_df['label'].value_counts())

        # Данные баланса.
        # Много способов сбалансировать данные... вы также можете попробовать назначить веса во время model.fit
        # Разделяем каждый класс, передискретизируем и объединяем обратно в один фрейм данных

        df_0 = skin_df[skin_df['label'] == 0]
        df_1 = skin_df[skin_df['label'] == 1]
        df_2 = skin_df[skin_df['label'] == 2]
        df_3 = skin_df[skin_df['label'] == 3]
        df_4 = skin_df[skin_df['label'] == 4]
        df_5 = skin_df[skin_df['label'] == 5]
        df_6 = skin_df[skin_df['label'] == 6]


        df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
        df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
        df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
        df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
        df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
        df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
        df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

        # Объединяем обратно в один фрейм данных
        skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced,
                                      df_2_balanced, df_3_balanced,
                                      df_4_balanced, df_5_balanced, df_6_balanced])

        # Проверить дистрибутив. Теперь все классы должны быть сбалансированы.
        # print(skin_df_balanced['label'].value_counts())

        # Теперь пришло время прочитать изображения на основе идентификатора изображения из CSV-файла.
        # Это самый безопасный способ чтения изображений, поскольку он гарантирует, что правильное изображение будет прочитано для правильного идентификатора
        image_path = {os.path.splitext(os.path.basename(x))[0]: x
                      for x in glob(os.path.join(self.dataset, '*', '*.jpg'))}

        # Определяем путь и добавляем как новый столбец
        skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)
        # Используйте путь для чтения изображений.
        skin_df_balanced['image'] = skin_df_balanced['path'].map(
            lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE))))
        print(skin_df_balanced)

        # Преобразование столбца dataframe изображений в массив numpy
        X = np.asarray(skin_df_balanced['image'].tolist())
        X = X / 255.  # Scale values to 0-1. You can also used standardscaler or other scaling methods.
        Y = skin_df_balanced['label']  # Assign label values to Y
        Y_cat = to_categorical(Y,num_classes=7)  # Convert to categorical as this is a multiclass classification problem
        # Разделить на обучение и тестирование
        x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)
        print(x_train)
        # Определить модель.
        # Я использовал autokeras, чтобы найти наилучшую модель для этой задачи.
        # Вы также можете загрузить предварительно обученные сети, такие как мобильная сеть или VGG16.

        num_classes = 7

        model = Sequential()
        model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
        # model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        # model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())

        model.add(Dense(32))
        model.add(Dense(7, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

        # Обучение
        # Вы также можете использовать генератор, чтобы использовать аугментацию во время обучения.

        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=2)

        score = model.evaluate(x_test, y_test)
        # print('Test accuracy:', score[1])

        model.save(f"models/model{SIZE}_imagesize{n_samples}_n_samples_{epochs}Epoch.h5")
        fig = plt.figure(figsize=(12, 8))
        # отображать точность обучения и проверки и потери в каждую эпоху
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(12, 8))
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Прогноз на тестовых данных
        y_pred = model.predict(x_test)
        # Преобразование классов прогнозов в один горячий вектор
        y_pred_classes = np.argmax(y_pred, axis=1)
        # Преобразование тестовых данных в один горячий вектор
        y_true = np.argmax(y_test, axis=1)

        # Вывести матрицу путаницы
        cm = confusion_matrix(y_true, y_pred_classes)

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.set(font_scale=1.6)
        sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

        # PLot дробно-неправильных классификаций
        incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
        plt.bar(np.arange(7), incorr_fraction)
        plt.xlabel('True Label')
        plt.ylabel('Fraction of incorrect predictions')


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

