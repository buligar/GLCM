import timeit

import cv2 # библиотека компьютерного зрения
import numpy # расширение языка Python, добавляющее поддержку больших многомерных массивов и матриц, вместе с большой библиотекой высокоуровневых математических функций для операций с этими массивами.
import matplotlib # библиотека для визуализации данных двумерной графикой
import os # предоставляет функции для взаимодействия с операционной системой
import shutil # модуль предлагает ряд высокоуровневых операций с файлами и коллекциями файлов.
import sys # обеспечивает доступ к некоторым переменным и функциям, взаимодействующим с интерпретатором python.
import np as np # все модули numpy и можете использовать их как np.
import numpy as np # все модули numpy и можете использовать их как np.
import pandas as pd # популярный инструментарий анализа данных
import matplotlib.pylab as plt # набор функций командного стиля, которые заставляют matplotlib работать как MATLAB.
import skimage.segmentation # модуль сегментации scikit-image
import skimage.filters.edges # модуль детектора границ
import tensorflow as tf
import pathlib
from PyQt5 import QtCore, QtWidgets, uic # QTCore - Модуль содержит основные классы, в том числе цикл событий и механизм сигналов и слотов Qt. Вспомогательные модули, для работы с виджетами и ui-фалйами, сгенерированными в дизайнере
from PyQt5.QtWidgets import QMenuBar, QMenu, QFileDialog, QGraphicsView, QApplication, QScrollArea, QGridLayout, \
    QWidget, QLabel, QScrollBar, QGraphicsScene  # подключение виджетов
from PyQt5.QtCore import QDir
from PyQt5.QtGui import QPixmap, QImage # подключение модуля для работы с изображениями
from cv2 import blur
from sklearn.pipeline import make_pipeline # подключение сокращения для конструктора конвейера
from skimage.color import rgb2lab # модуль для преобразования из rgb в lab
from sklearn.preprocessing import StandardScaler # подключение преобразователя данных
from skimage.feature import greycomatrix, greycoprops # подключение модуля для построения МПС и вычисления текстурных признаков
from sklearn.svm import SVC # подключения классификатора SVC
from matplotlib import pyplot as plt # Pyplot предоставляет интерфейс конечного автомата для базовой библиотеки построения графиков в matplotlib.
from tkinter import * # пакет для Python, предназначенный для работы с библиотекой Tk
from PIL import Image, ImageDraw
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Sequential




matplotlib.use('QT5Agg')

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self): # Инициализация
        super(MyWindow, self).__init__()
        self.ui=uic.loadUi('GLCM.ui', self) # Импорт интерфейса
        self.addFunctions() # Вызов функций


    def addFunctions(self):
        # Menu
        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)
        fileMenu = QMenu("&Файл", self)
        self.menuBar.addMenu(fileMenu)
        fileMenu.addAction('Открыть',self.action_clicked)
        fileMenu.addAction('Сохранить', self.action_clicked)
        fileMenu.addAction('Загрузить обучающую выборку', self.action_clicked)
        fileMenu.addAction('Загрузить тестовую выборку', self.action_clicked)
        self.listWidget.itemClicked.connect(self.listitemclicked) # load image
        self.pushButton_deletehair.clicked.connect(lambda: self.removehair())
        self.pushButton_segmentation.clicked.connect(lambda: self.segmentation())  # Сегментация
        self.pushButton.clicked.connect(lambda: self.glcm()) # GLCM
        self.pushButton_deleteBackground.clicked.connect(lambda: self.deletebackground()) # Удаление фона
        self.Buttontest.clicked.connect(lambda: self.test()) # Тестовая функция
        self.info_button.clicked.connect(lambda: self.informativ()) # Информативность
        self.info_button_2.clicked.connect(lambda: self.informativ2())
        self.zoom_in_button.clicked.connect(lambda: self.on_zoom_in()) # Увеличить
        self.zoom_out_button.clicked.connect(lambda: self.on_zoom_out()) # Уменьшить
        self.SVM_button.clicked.connect(lambda: self.SVM()) # SVM Классификатор
        self.pushButton_2.clicked.connect(lambda: self.glcm_all())
        # self.KNN_button.clicked.connect(lambda: self.KNN()) # KNN Классификатор
        # self.NN_button.clicked.connect(lambda: self.NN())

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
        elif action.text() == "Сохранить":
            fname = QFileDialog.getSaveFileName(self)[0]
            try:
                f = open(fname, 'w')
                text = self.text_edit.toPlainText()
                f.write(text)
                f.close()
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


    def listitemclicked(self): # Выбор из списка изображения и отображение на label

        self.file_path = self.listWidget.selectedItems()[0].text()  # Путь к выбранному файлу
        # for index, link in enumerate(self.files_path):
        #     if self.file_path == link:
        #         self.file_index=index

        self.scene = QGraphicsScene(self.graphicsView)
        self.pixmap = QPixmap(self.file_path)
        self.scene.addPixmap(self.pixmap)
        self.graphicsView.setScene(self.scene)
        self.scene_2 = QGraphicsScene(self.graphicsView_2)
        self.pixmap2 = QPixmap(self.file_path)
        self.scene_2.addPixmap(self.pixmap2)
        self.graphicsView_2.setScene(self.scene_2)

        self.scale = 1


    def on_zoom_in(self): # Увеличение изображения
        self.scene.clear()
        self.scene_2.clear()
        self.scale *= 2
        self.resize_image()

    def on_zoom_out(self): # Уменьшение изображения
        self.scene.clear()
        self.scene_2.clear()
        self.scale /= 2
        self.resize_image()

    def resize_image(self): # Изменение изображения
        size = self.pixmap.size()
        scaled_pixmap = self.pixmap.scaled(self.scale * size)
        self.scene.addPixmap(scaled_pixmap)
        size = self.pixmap2.size()
        scaled_pixmap = self.pixmap2.scaled(self.scale * size)
        self.scene_2.addPixmap(scaled_pixmap)
    def removehair(self):


        image = cv2.imread(self.file_path)
        image_resize = cv2.resize(image, (1800, 1200))
        # plt.subplot(2,3,1)
        # Convert the original image to grayscale
        # plt.imshow(cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.title('Original : ' )

        grayScale = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)
        # plt.subplot(2,3,2)
        # plt.imshow(grayScale)
        # plt.axis('off')
        # plt.title('GrayScale : ' )

        # Kernel for the morphological filtering
        kernel = cv2.getStructuringElement(1, (17, 17))

        # Perform the blackHat filtering on the grayscale image to find the hair countours
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        # plt.subplot(2,3,3)
        # plt.imshow(blackhat)
        # plt.axis('off')
        # plt.title('blackhat : ' )

        # intensify the hair countours in preparation for the inpainting
        ret, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        # plt.subplot(2,3,4)
        # plt.imshow(threshold)
        # plt.axis('off')
        # plt.title('threshold : ')

        # inpaint the original image depending on the mask
        final_image = cv2.inpaint(image_resize, threshold, 1, cv2.INPAINT_TELEA)
        # plt.subplot(2,3,5)
        # plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        cv2.imwrite("withouthair.png", final_image)
        self.scene_2 = QGraphicsScene(self.graphicsView_2)
        self.pixmap2 = QPixmap('withouthair.png')
        self.scene_2.addPixmap(self.pixmap2)
        self.file_path = 'withouthair.png'
        # plt.axis('off')
        # plt.title('final_image : ' )
        # plt.plot()
        # plt.show()

    def segmentation(self): # Сегментация


        # img = cv2.imread('withouthair.png')
        # gray = cv2.imread('withouthair.png',0)
        img = cv2.imread(self.file_path)
        gray = cv2.imread(self.file_path,0)


        ## (2) Threshold
        th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )

        ## (3) Find the first contour greater than 100 located in the central area
        cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea)
        # Клетки
        # H, W = img.shape[:2]
        # for cnt in cnts:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     if cv2.contourArea(cnt) > 100 and (1 < w / h < 2) and (W / 2 < x + w // 2 < W * 1 / 2) and (H / 1 < y + h // 2 < H * 1 / 2):
        #         break

        H, W = img.shape[:2]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 100 and (1 < w / h < 2) and (W / 2 < x + w // 2 < W * 1 / 2) and (H / 1 < y + h // 2 < H * 1 / 2):
                break

        ## (4) Create mask and do bitwise-op
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=mask)

        ## Display it
        cv2.imwrite("dst.png", dst)
        self.scene_2 = QGraphicsScene(self.graphicsView_2)
        self.pixmap2 = QPixmap('dst.png')
        self.scene_2.addPixmap(self.pixmap2)
        self.graphicsView_2.setScene(self.scene_2)
        self.file_path='dst.png'
        # self.scene_2.clear()

        # ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # # Otsu's thresholding
        # ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # Otsu's thresholding after Gaussian filtering
        # blur = cv2.GaussianBlur(img, (5, 5), 0)
        # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # plot all the images and their histograms
        # images = [img, 0, th1,
        #           img, 0, th2,
        #           blur, 0, th3]
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



    def deletebackground(self): # Удаления фона

        img = cv2.imread(self.file_path)
        # print(img)

        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # apply morphology to clean small spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        # get external contour
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)

        # draw white filled contour on black background as mas
        contour = np.zeros_like(gray)
        cv2.drawContours(contour, [big_contour], 0, 255, -1)

        # blur dilate image
        blur = cv2.GaussianBlur(contour, (5, 5), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

        # stretch so that 255 -> 255 and 127.5 -> 0
        mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5, 255), out_range=(0, 255))

        # put mask into alpha channel of input
        self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        self.result[:, :, 3] = mask
        # print(self.result)
        # save output
        cv2.imwrite('withoutBackground.png', self.result)

        # Display various images to see the steps
        self.scene_2.clear()
        self.pixmap2 = QPixmap('withoutBackground.png')
        self.scene_2.addPixmap(self.pixmap2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def glcm(self):

        # Очищение файла glcm_all.csv
        my_file = open("glcm_all.csv", "w+")
        my_file.close()
        file_csv_path = 'glcm_all.csv'
        os.remove(file_csv_path)
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'texture')
        shutil.rmtree(path)
        os.mkdir("texture")
        np.set_printoptions(edgeitems=10000)  # для отображения полной матрицы изображения в окне вывода
        self.features_sum = 24 * self.spinBox.value() # Всего признаков
        head = np.array(range(0,self.features_sum+1)).flatten() # Массив кол-ва признаков
        num = len(self.files_path) # Всего изображений
        self.num_cells=np.array(range(1,num+1)).flatten() # Массив кол-ва изображений
        mat = np.matrix(head)
        df = pd.DataFrame(mat)
        df.to_csv('glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';') # Указание кол-ва признаков

        for i in range(num):
            self.file_path=self.files_path[i]
            print(self.file_path)
            # self.scene = QGraphicsScene(self.graphicsView)
            # self.pixmap = QPixmap(self.file_path)
            # self.scene.addPixmap(self.pixmap)
            # self.graphicsView.setScene(self.scene)
            # self.removehair()
            self.segmentation()
            np.set_printoptions(edgeitems=1000) # для отображения полной матрицы изображения в окне вывода
            self.file_path = 'dst.png'
            self.grayscale = cv2.imread(self.file_path)  # Преобразование в оттенки серого
            color = self.ui.color.currentIndex()
            # Сохранение и отображение канала цвета
            if color == 0:
                self.grayscale[:, :, 0] = 0
                self.grayscale[:, :, 1] = 0
                cv2.imwrite('r.bmp', self.grayscale)
                self.scene_2.clear()
                self.pixmap2 = QPixmap('r.bmp')
                self.scene_2.addPixmap(self.pixmap2)
            elif color == 1:
                self.grayscale[:, :, 0] = 0
                self.grayscale[:, :, 2] = 0
                cv2.imwrite('g.bmp', self.grayscale)
                self.scene_2.clear()
                self.pixmap2 = QPixmap('g.bmp')
                self.scene_2.addPixmap(self.pixmap2)
            elif color == 2:
                self.grayscale[:, :, 1] = 0
                self.grayscale[:, :, 2] = 0
                cv2.imwrite('b.bmp', self.grayscale)
                self.scene_2.clear()
                self.pixmap2 = QPixmap('b.bmp')
                self.scene_2.addPixmap(self.pixmap2)
            elif color == 3:
                self.grayscale = cv2.imread(self.file_path, 0)
                cv2.imwrite('gray.bmp', self.grayscale)
                self.scene_2.clear()
                self.pixmap2 = QPixmap('gray.bmp')
                self.scene_2.addPixmap(self.pixmap2)

            # Извлечение канала цвета
            self.grayscale=cv2.imread(self.file_path) # Преобразование в оттенки серого
            if color == 0:
                self.grayscale=self.grayscale[:, :, 2]
            elif color == 1:
                self.grayscale=self.grayscale[:, :, 1]
            elif color == 2:
                self.grayscale=self.grayscale[:, :, 0]
            elif color == 3:
                self.grayscale = cv2.imread(self.file_path,0)

            self.value = self.spinBox.value() # Значение Расстояние смежности в spinbox
            Distances = np.arange(1, self.value + 1, 1) # Расстояние смежности на графике
            glcm = greycomatrix(self.grayscale, distances=Distances,
                                angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
                                levels=256,
                                symmetric=True, normed=True)  # Построение МПС
            # print(glcm)
            filt_glcm = glcm[1:, 1:, :, :] # Не берет в расчет пиксель 0

            Contrast =  greycoprops(filt_glcm, 'contrast')  # Текстурный признак Контраст
            Dissimilarity = greycoprops(filt_glcm, 'dissimilarity')  # Текстурный признак несходство
            Homogeneity = greycoprops(filt_glcm, 'homogeneity')  # Текстурный признак Локальная однородность
            Asm = greycoprops(filt_glcm, 'ASM')  # Текстурный признак Угловой второй момент
            Energy = greycoprops(filt_glcm, 'energy')  # Текстурный признак Энергия
            Correlation = greycoprops(filt_glcm, 'correlation')  # Текстурный признак Корреляция

            # из двумерного массива в одномерный
            self.Contrast = np.concatenate(Contrast)
            self.Dissimilarity = np.concatenate(Dissimilarity)
            self.Homogeneity = np.concatenate(Homogeneity)
            self.Asm = np.concatenate(Asm)
            self.Energy = np.concatenate(Energy)
            self.Correlation = np.concatenate(Correlation)

            # Сохранение по отдельности
            Contrast_all = np.transpose(Contrast)
            Contrast0 = Contrast_all[0]
            Contrast45 = Contrast_all[1]
            Contrast90 = Contrast_all[2]
            Contrast135 = Contrast_all[3]
            with open('texture\Contrast0.csv', 'a') as f:
                np.savetxt(f, Contrast0,delimiter=';', fmt='%.5f')
            with open('texture\Contrast45.csv', 'a') as f:
                np.savetxt(f, Contrast45,delimiter=';', fmt='%.5f')
            with open('texture\Contrast90.csv', 'a') as f:
                np.savetxt(f, Contrast90,delimiter=';', fmt='%.5f')
            with open('texture\Contrast135.csv', 'a') as f:
                np.savetxt(f, Contrast135,delimiter=';', fmt='%.5f')

            Dissimilarity_all = np.transpose(Dissimilarity)
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

            Homogeneity_all = np.transpose(Homogeneity)
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


            Asm_all = np.transpose(Asm)
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

            Energy_all = np.transpose(Energy)
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

            Correlation_all = np.transpose(Correlation)
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

            # Сохранение все вместе
            self.GLCM_All =[self.Contrast] + [self.Dissimilarity] + [self.Homogeneity] + [self.Asm] + [self.Energy] + [self.Correlation]
            self.GLCM_All=np.concatenate(self.GLCM_All)
            mat = np.matrix(self.GLCM_All)
            con = np.matrix(self.Contrast)
            dis = np.matrix(self.Dissimilarity)
            Hom = np.matrix(self.Homogeneity)
            Asm2 = np.matrix(self.Asm)
            Eng = np.matrix(self.Energy)
            Corr = np.matrix(self.Correlation)
            df = pd.DataFrame(mat)
            df1 = pd.DataFrame(con)
            df2 = pd.DataFrame(dis)
            df3 = pd.DataFrame(Hom)
            df4 = pd.DataFrame(Asm2)
            df5 = pd.DataFrame(Eng)
            df6 = pd.DataFrame(Corr)
            df.to_csv('glcm_all.csv',mode='a',header=False,index=[1,2,3,4,5], float_format="%.5f", sep=';')
            df1.to_csv('texture\Contrast.csv', mode='a', header=False, index=[1, 2, 3, 4, 5], float_format="%.5f", sep=';')
            df2.to_csv('texture\Dissimilarity.csv', mode='a', header=False, index=[1, 2, 3, 4, 5], float_format="%.5f", sep=';')
            df3.to_csv('texture\Homogeneity.csv', mode='a', header=False, index=[1, 2, 3, 4, 5], float_format="%.5f", sep=';')
            df4.to_csv('texture\Asm.csv', mode='a', header=False, index=[1, 2, 3, 4, 5], float_format="%.5f", sep=';')
            df5.to_csv('texture\Energy.csv', mode='a', header=False, index=[1, 2, 3, 4, 5], float_format="%.5f", sep=';')
            df6.to_csv('texture\Correlation.csv', mode='a', header=False, index=[1, 2, 3, 4, 5], float_format="%.5f", sep=';')
            self.scene_2 = QGraphicsScene(self.graphicsView_2)
            self.pixmap2 = QPixmap(self.file_path)
            self.scene_2.addPixmap(self.pixmap2)
            self.graphicsView_2.setScene(self.scene_2)


        #Отображение на графиках
        plt.clf()
        plt.subplot(2, 3, 1)
        plt.grid(axis='both')
        plt.title("Контраст")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Contrast, marker='o')

        plt.subplot(2, 3, 2)
        plt.grid(axis='both')
        plt.title("Несходство")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Dissimilarity, marker='o')

        plt.subplot(2, 3, 3)
        plt.grid(axis='both')
        plt.title("Локальная однородность")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Homogeneity, marker='o')

        plt.subplot(2, 3, 4)
        plt.grid(axis='both')
        plt.title("Угловой второй момент")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Asm, marker='o')

        plt.subplot(2, 3, 5)
        plt.grid()
        plt.title("Энергия")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Energy, marker='o')

        plt.subplot(2, 3, 6)
        plt.grid(axis='both')
        plt.title("Корреляция")
        plt.xticks([i for i in range(0, max(Distances + 1))])
        plt.plot(Distances, Correlation, marker='o')


        self.Anglelegend = [0, 45, 90, 135]

        plt.figlegend(self.Anglelegend)
        plt.show()
        self.features_describe()

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
            self.glcm()
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
            self.glcm()
            file_oldname = os.path.join(self.pred_test_papka, "glcm_all.csv")
            file_newname_newfile = os.path.join(self.pred_test_papka, f"glcm_test{k}.csv")
            os.rename(file_oldname, file_newname_newfile)


    def features_describe(self): # Сохранение описания значений таблицы (mean,std)
        features = pd.read_csv('glcm_all.csv', delimiter=';')
        describe = features.describe()
        df = pd.DataFrame(describe)
        df.to_csv('glcm_describe.csv')


    def test(self): # Тестовая функция
        np.set_printoptions(edgeitems=100000)
        print(self.file_path)
        image=cv2.imread(self.file_path) # получаем матрицу изображения BGR
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # конвертируем из BGR в GRAY
        img = rgb2lab(img)
        # img = np.array(img, dtype=np.uint8)
        print(img)
        # glcm = greycomatrix(img, [1], [0], 256, symmetric=True, normed=False)
        # filt_glcm = glcm[1:, 1:, :, :]
        # # for i in filt_glcm:
        # #     print(*i)
        # dissimilarity = greycoprops(filt_glcm, 'dissimilarity')[0][0]
        # correlation = greycoprops(filt_glcm, 'correlation')[0][0]
        # homogeneity = greycoprops(filt_glcm, 'homogeneity')[0][0] # однородность
        # energy = greycoprops(filt_glcm, 'energy')[0][0] #
        # del img
        # feature = np.array([dissimilarity, correlation, homogeneity, energy])
        # print(feature)
        # return feature


    def informativ(self): # вычисление и отображение среднего значения, среднеквадратического значения и информативности в общем
        np.set_printoptions(edgeitems=100000)
        # Проверка на указание расстояния смежности
        if self.spinBox.value()==0:
            print("Укажите расстояние смежности в spinbox")
        self.features_sum = 24 * self.spinBox.value()
        self.rass = self.spinBox.value()  # Расстояние смежности
        self.rassipriznaki = 6 * self.rass  # Расстояние смежности * 6 признаков
        # Загрузка данных
        features = pd.read_csv('glcm_train0.csv',delimiter=';')
        del features['0']
        mean1=features.describe().loc[['mean']]
        std1=features.describe().loc[['std']]
        features = pd.read_csv('glcm_train1.csv', delimiter=';')
        del features['0']
        mean2 = features.describe().loc[['mean']]
        std2 = features.describe().loc[['std']]
        # Расчет информативности
        c=numpy.array(abs(mean1-mean2))
        z=numpy.array(1.6*(std1+std2))
        self.informativeness=numpy.divide(c,z)
        infocsv=self.informativeness.reshape(4,self.rassipriznaki)
        df = pd.DataFrame(infocsv)
        df.to_csv('informativness.csv', header=False, index=[1, 2, 3, 4, 5], float_format="%.5f", sep=';')
        head = np.arange(1,self.features_sum+1,1)
        self.mean1=numpy.array(mean1)
        self.mean2 = numpy.array(mean2)
        self.std1=numpy.array(std1)
        self.std2=numpy.array(std2)

        #Отоброжение на графиках
        # plt.subplot(2, 3, 1)
        # plt.grid(axis='both')
        # plt.title("Зависимость среднего 1 класса от расстояния смежности")
        # plt.plot(head, self.mean1[0], marker='o')
        # plt.show()
        # plt.subplot(2, 3, 2)
        # plt.grid(axis='both')
        # plt.title("Зависимость среднего 2 класса от расстояния смежности")
        # plt.plot(head, self.mean2[0], marker='o')
        # plt.show()
        # plt.subplot(2, 3, 4)
        # plt.grid(axis='both')
        # plt.title("Зависимость среднеквадратического отклонения 1 класса\n от расстояния смежности")
        # plt.plot(head, self.std1[0], marker='o')
        # plt.show()
        # plt.subplot(2, 3, 5)
        # plt.grid(axis='both')
        # plt.title("Зависимость среднеквадратического отклонения 2 класса\n от расстояния смежности")
        # plt.plot(head, self.std2[0], marker='o')
        # plt.show()
        x=4*self.spinBox.value()
        ar1=np.arange(0,x)
        ar2=np.arange(x,2*x)
        ar3=np.arange(2*x,3*x)
        ar4=np.arange(3*x,4*x)
        ar5=np.arange(4*x,5*x)
        ar6=np.arange(5*x,6*x)
        df1 = self.informativeness[0][ar1]
        df2 = self.informativeness[0][ar2]
        df3 = self.informativeness[0][ar3]
        df4 = self.informativeness[0][ar4]
        df5 = self.informativeness[0][ar5]
        df6 = self.informativeness[0][ar6]
        arange1 = np.arange(1, x+1)
        arange2 = np.arange(x+1, 2 * x+1)
        arange3 = np.arange(2 * x+1, 3 * x+1)
        arange4 = np.arange(3 * x+1, 4 * x+1)
        arange5 = np.arange(4 * x+1, 5 * x+1)
        arange6 = np.arange(5 * x+1, 6 * x+1)

        # plt.subplot(2, 3)
        plt.grid(axis='both')
        plt.title("Зависимость информативности от расстояния смежности")
        plt.plot(arange1, df1,marker='o',color='red')
        plt.plot(arange2, df2,marker='o',color='green')
        plt.plot(arange3, df3, marker='o', color='blue')
        plt.plot(arange4, df4, marker='o', color='yellow')
        plt.plot(arange5, df5, marker='o', color='orange')
        plt.plot(arange6, df6, marker='o', color='purple')

        self.Anglelegend = ['Контраст','Несходство','Локальная однородность','Угловой второй момент','Энергия','Корреляция']
        plt.figlegend(self.Anglelegend)
        plt.show()


    def informativ2(self): # Вычисление и отображение информативности признаков по отдельности
        np.set_printoptions(edgeitems=100000)
        # Проверка на указание расстояния смежности
        if self.spinBox.value() == 0:
            print("Укажите расстояние смежности в spinbox")
        self.features_sum = 24 * self.spinBox.value() # сколько всего признаков
        self.rass= self.spinBox.value() # Расстояние смежности
        self.rassinapravlenie=4*self.rass # Расстояние смежности * 4 направления
        # Загрузка данных
        features = pd.read_csv('glcm_train0.csv', delimiter=';')
        del features['0']
        print(features)
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
        infoContrast = np.transpose(self.informativeness.reshape(6, self.rassinapravlenie)[0].reshape(self.rass,4))
        maxContrast=max(map(max, infoContrast))
        infoDissimilation = np.transpose(self.informativeness.reshape(6, self.rassinapravlenie)[1].reshape(self.rass, 4))
        maxDissimilation = max(map(max, infoDissimilation))
        infoHomogeneity = np.transpose(self.informativeness.reshape(6, self.rassinapravlenie)[2].reshape(self.rass, 4))
        maxHomogeneity = max(map(max, infoHomogeneity))
        infoAsm = np.transpose(self.informativeness.reshape(6, self.rassinapravlenie)[3].reshape(self.rass, 4))
        maxAsm = max(map(max, infoAsm))
        infoEnergy = np.transpose(self.informativeness.reshape(6, self.rassinapravlenie)[4].reshape(self.rass, 4))
        maxEnergy = max(map(max, infoEnergy))
        infoCorrelation = np.transpose(self.informativeness.reshape(6, self.rassinapravlenie)[5].reshape(self.rass, 4))
        maxCorrelation = max(map(max, infoCorrelation))
        rass_zmez=np.arange(1,self.spinBox.value()+1,1)

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
        self.Anglelegend = [0, 45, 90, 135]

        plt.figlegend(self.Anglelegend)
        plt.show()

    def SVM(self): # Классификатор опорных векторов

        # Загрузка обучающей выборки
        cell1 = pd.read_csv("glcm_train0.csv", delimiter=';')
        cell1 = cell1.drop(columns=cell1.columns[0])
        cell1["class"] = 0
        cell2 = pd.read_csv("glcm_train1.csv", delimiter=';')
        cell2 = cell2.drop(columns=cell2.columns[0])
        cell2["class"] = 1
        cells = pd.concat([cell1,cell2])
        cells = cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
        print(cells)

        # Загрузка тестовой выборки
        test_cell1 = pd.read_csv("glcm_test0.csv", delimiter=';')
        test_cell1 = test_cell1.drop(columns=test_cell1.columns[0])
        test_cell1["class"] = 0
        test_cell2 = pd.read_csv("glcm_test1.csv", delimiter=';')
        test_cell2 = test_cell2.drop(columns=test_cell2.columns[0])
        test_cell2["class"] = 1
        test_cells = pd.concat([test_cell1, test_cell2])
        test_cells = test_cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))

        # Соединение и преобразование выборок
        x_train = cells.drop(columns=cells.columns[-1]).to_numpy()
        y_train = cells.iloc[:, -1:].to_numpy().flatten()
        x_test = test_cells.drop(columns=test_cells.columns[-1]).to_numpy()
        y_test = test_cells.iloc[:, -1:].to_numpy().flatten()

        #Настройки ядра
        svc_rbf = SVC(C=1, kernel='rbf', cache_size=1000)

        # Обучение
        svc_rbf_pipe = make_pipeline(StandardScaler(), svc_rbf)
        svc_rbf_pipe.fit(x_train, y_train)

        # Тестирование
        results = svc_rbf_pipe.score(x_test, y_test)
        print(results)

        self.ui.accuracy.setNum(results) # отображение точности
        self.num_test = self.spinBox_2.value()-1 # Номер исследуемого изображения
        res = svc_rbf_pipe.predict([x_test[self.num_test]])  # предсказание класса для выбранного изображения из тестовой выборки
        self.ui.results.setNum(res[0]+1) # отображение номера класса
        self.show() # показать на интерфейсе все значения

    # def KNN(self): # Классификатор k-ближайших соседей
    #
    #     # Загрузка обучающей выборки
    #     blasts = pd.read_csv("glcm_train1.csv", delimiter=';')
    #     blasts = blasts.drop(columns=blasts.columns[0])
    #     blasts["class"] = 0
    #     lymphocytes = pd.read_csv("glcm_train2.csv", delimiter=';')
    #     lymphocytes = lymphocytes.drop(columns=lymphocytes.columns[0])
    #     lymphocytes["class"] = 1
    #     cells = pd.concat([blasts, lymphocytes])
    #     cells = cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    #     print(cells)
    #
    #     # Загрузка тестовой выборки
    #     test_blasts = pd.read_csv("glcm_test1.csv", delimiter=';')
    #     test_blasts = test_blasts.drop(columns=test_blasts.columns[0])
    #     test_blasts["class"] = 0
    #     test_lymphocytes = pd.read_csv("glcm_test2.csv", delimiter=';')
    #     test_lymphocytes = test_lymphocytes.drop(columns=test_lymphocytes.columns[0])
    #     test_lymphocytes["class"] = 1
    #     test_cells = pd.concat([test_blasts, test_lymphocytes])
    #     test_cells = test_cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    #
    #     # Соединение и преобразование выборок
    #     x_train = cells.drop(columns=cells.columns[-1]).to_numpy()
    #     y_train = cells.iloc[:, -1:].to_numpy().flatten()
    #     x_test = test_cells.drop(columns=test_cells.columns[-1]).to_numpy()
    #     y_test = test_cells.iloc[:, -1:].to_numpy().flatten()
    #
    #     results = {}
    #     for i in range(100):
    #         neigh_pipe = make_pipeline(
    #             StandardScaler(),
    #             KNeighborsClassifier(n_neighbors=i + 2)
    #         )
    #         neigh_pipe.fit(x_train, y_train)
    #         results[i] = neigh_pipe.score(x_test, y_test)
    #     acc = 0.001
    #     n_neighbors = 0
    #     for k, v in results.items():
    #         if v > acc:
    #             acc = v
    #             n_neighbors = k
    #     print("Средняя точность по тестовой выборке:", acc)
    #     print("Оптимальное количество соседей:", n_neighbors)
    #
    #     self.ui.accuracy.setNum(acc)  # отображение точности
    #     self.num_test = self.spinBox_2.value() - 1  # Номер исследуемого изображения
    #     res = neigh_pipe.predict([x_test[self.num_test]])  # предсказание класса для выбранного изображения из тестовой выборки
    #     self.ui.results.setNum(res[0] + 1)  # отображение номера класса
    #     self.show()  # показать на интерфейсе все значения

    # def ISIC(self):
    #     train = pd.read_csv('ISIC_train.csv')
    #     test = pd.read_csv('ISIC_test.csv')
    #     print('Train: ', train.shape)
    #     print("Test:", test.shape)
    #     print(train.head())
    #     print(train.info())
    #     print(test.head())
    #     print(test.info())
    #     print(train['benign_malignant'].value_counts(normalize=True))
    #     a=train.sort_values(by = 'benign_malignant', ascending=False)
    #     a=a[a.target == 1]
    #     print(a)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

