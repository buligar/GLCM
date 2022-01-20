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
from PyQt5 import QtCore, QtWidgets, uic # QTCore - Модуль содержит основные классы, в том числе цикл событий и механизм сигналов и слотов Qt. Вспомогательные модули, для работы с виджетами и ui-фалйами, сгенерированными в дизайнере
from PyQt5.QtWidgets import QMenuBar, QMenu, QFileDialog, QApplication, QScrollArea, QGridLayout, QWidget, QLabel, QScrollBar # подключение виджетов
from PyQt5.QtGui import QPixmap, QImage # подключение модуля для работы с изображениями
from sklearn.pipeline import make_pipeline # подключение сокращения для конструктора конвейера
from skimage.color import rgb2lab # модуль для преобразования из rgb в lab
from sklearn.preprocessing import StandardScaler # подключение преобразователя данных
from skimage.feature import greycomatrix, greycoprops # подключение модуля для построения МПС и вычисления текстурных признаков
from sklearn.svm import SVC # подключения классификатора SVC
from matplotlib import pyplot as plt # Pyplot предоставляет интерфейс конечного автомата для базовой библиотеки построения графиков в matplotlib.
from tkinter import * # пакет для Python, предназначенный для работы с библиотекой Tk

import tensorflow as tf
import pathlib
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
        self.listWidget.itemClicked.connect(self.listitemclicked) # load image
        self.pushButton.clicked.connect(lambda: self.glcm()) # GLCM
        self.pushButton_segmentation.clicked.connect(lambda: self.segmentation()) # Сегментация
        self.pushButton_deleteBackground.clicked.connect(lambda: self.deletebackground()) # Удаление фона
        self.Buttontest.clicked.connect(lambda: self.test()) # Тестовая функция
        self.info_button.clicked.connect(lambda: self.informativ()) # Информативность
        self.info_button_2.clicked.connect(lambda: self.informativ2())
        self.zoom_in_button.clicked.connect(lambda: self.on_zoom_in()) # Увеличить
        self.zoom_out_button.clicked.connect(lambda: self.on_zoom_out()) # Уменьшить
        self.SVM_button.clicked.connect(lambda: self.SVM()) # SVM Классификатор
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

    def listitemclicked(self): # Выбор из списка изображения и отображение на label

        self.file_path = self.listWidget.selectedItems()[0].text()  # Путь к выбранному файлу
        for index, link in enumerate(self.files_path):
            if self.file_path == link:
                self.file_index=index

        # Отображение изображения на label
        self.pixmap=QPixmap(self.file_path)
        self.ui.label.setPixmap(self.pixmap)
        self.ui.label.repaint()
        self.pixmap2 = QPixmap(self.file_path)
        self.ui.dst.setPixmap(self.pixmap2)
        self.ui.dst.repaint()

        self.scale = 1
        print(self.file_path)

    def on_zoom_in(self): # Увеличение изображения
        self.scale *= 2
        self.resize_image()

    def on_zoom_out(self): # Уменьшение изображения
        self.scale /= 2
        self.resize_image()

    def resize_image(self): # Изменение изображения
        size = self.pixmap.size()
        scaled_pixmap = self.pixmap.scaled(self.scale * size)
        self.label.setPixmap(scaled_pixmap)
        size = self.pixmap2.size()
        scaled_pixmap = self.pixmap2.scaled(self.scale * size)
        self.dst.setPixmap(scaled_pixmap)


    def segmentation(self): # Сегментация

        img = cv2.imread(self.file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ## (2) Threshold
        th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        ## (3) Find the first contour that greate than 100, locate in centeral region
        ## Adjust the parameter when necessary
        cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea)
        H, W = img.shape[:2]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 100 and (1 < w / h < 2) and (W / 2 < x + w // 2 < W * 1 / 2) and (
                    H / 1 < y + h // 2 < H * 1 / 2):
                break

        ## (4) Create mask and do bitwise-op
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=mask)

        ## Display it
        cv2.imwrite("dst.png", dst)
        self.pixmap2 = QPixmap('dst.png')
        self.ui.dst.setPixmap(self.pixmap2)
        self.ui.dst.repaint()


    def deletebackground(self): # Удаления фона

        img = cv2.imread(self.file_path)
        print(img)

        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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
        print(self.result)
        # save output
        cv2.imwrite('withoutBackground.png', self.result)

        # Display various images to see the steps
        pixmap = QPixmap('withoutBackground.png')
        self.ui.label.setPixmap(pixmap)
        self.ui.label.repaint()
        # cv2.imshow('result', self.result)

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
            self.file_path = self.files_path[i] # путь
            # self.deletebackground()
            np.set_printoptions(edgeitems=1000) # для отображения полной матрицы изображения в окне вывода
            self.grayscale = cv2.imread(self.file_path)  # Преобразование в оттенки серого
            color = self.ui.color.currentIndex()
            # Сохранение и отображение канала цвета
            if color == 0:
                self.grayscale[:, :, 0] = 0
                self.grayscale[:, :, 1] = 0
                cv2.imwrite('r.bmp', self.grayscale)
                self.pixmap2 = QPixmap('r.bmp')
                self.ui.dst.setPixmap(self.pixmap2)
                self.ui.dst.repaint()
            elif color == 1:
                self.grayscale[:, :, 0] = 0
                self.grayscale[:, :, 2] = 0
                cv2.imwrite('g.bmp', self.grayscale)
                self.pixmap2 = QPixmap('g.bmp')
                self.ui.dst.setPixmap(self.pixmap2)
                self.ui.dst.repaint()
            elif color == 2:
                self.grayscale[:, :, 1] = 0
                self.grayscale[:, :, 2] = 0
                cv2.imwrite('b.bmp', self.grayscale)
                self.pixmap2 = QPixmap('b.bmp')
                self.ui.dst.setPixmap(self.pixmap2)
                self.ui.dst.repaint()
            elif color == 3:
                self.grayscale = cv2.imread(self.file_path, 0)
                cv2.imwrite('gray.bmp', self.grayscale)
                self.pixmap2 = QPixmap('gray.bmp')
                self.ui.dst.setPixmap(self.pixmap2)
                self.ui.dst.repaint()

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



        #Отображение на графиках
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
        features = pd.read_csv('glcm_train1.csv',delimiter=';')
        del features['0']
        print(features)
        mean1=features.describe().loc[['mean']]
        std1=features.describe().loc[['std']]
        features = pd.read_csv('glcm_train2.csv', delimiter=';')
        del features['0']
        mean2 = features.describe().loc[['mean']]
        std2 = features.describe().loc[['std']]
        # Расчет информативности
        c=numpy.array(abs(mean1-mean2))
        z=numpy.array(1.6*(std1+std2))
        self.informativeness=numpy.divide(c,z)
        print(self.informativeness)
        infocsv=self.informativeness.reshape(4,self.rassipriznaki)
        df = pd.DataFrame(infocsv)
        df.to_csv('informativness.csv', header=False, index=[1, 2, 3, 4, 5], float_format="%.5f", sep=';')
        head = np.arange(1,self.features_sum+1,1)
        self.mean1=numpy.array(mean1)
        self.mean2 = numpy.array(mean2)
        self.std1=numpy.array(std1)
        self.std2=numpy.array(std2)

        #Отоброжение на графиках
        plt.subplot(2, 3, 1)
        plt.grid(axis='both')
        plt.title("Зависимость среднего 1 класса от расстояния смежности")
        plt.plot(head, self.mean1[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 2)
        plt.grid(axis='both')
        plt.title("Зависимость среднего 2 класса от расстояния смежности")
        plt.plot(head, self.mean2[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 4)
        plt.grid(axis='both')
        plt.title("Зависимость среднеквадратического отклонения 1 класса\n от расстояния смежности")
        plt.plot(head, self.std1[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 5)
        plt.grid(axis='both')
        plt.title("Зависимость среднеквадратического отклонения 2 класса\n от расстояния смежности")
        plt.plot(head, self.std2[0], marker='o')
        plt.show()
        plt.subplot(2, 3, 6)
        plt.grid(axis='both')
        plt.title("Зависимость информативности от расстояния смежности")
        plt.plot(head, self.informativeness[0],marker='o')
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
        features = pd.read_csv('glcm_train1.csv', delimiter=';')
        del features['0']
        print(features)
        mean1 = features.describe().loc[['mean']]
        std1 = features.describe().loc[['std']]
        features = pd.read_csv('glcm_train2.csv', delimiter=';')
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
        blasts = pd.read_csv("glcm_train1.csv", delimiter=';')
        blasts = blasts.drop(columns=blasts.columns[0])
        blasts["class"] = 0
        lymphocytes = pd.read_csv("glcm_train2.csv", delimiter=';')
        lymphocytes = lymphocytes.drop(columns=lymphocytes.columns[0])
        lymphocytes["class"] = 1
        cells = pd.concat([blasts, lymphocytes])
        cells = cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
        print(cells)

        # Загрузка тестовой выборки
        test_blasts = pd.read_csv("glcm_test1.csv", delimiter=';')
        test_blasts = test_blasts.drop(columns=test_blasts.columns[0])
        test_blasts["class"] = 0
        test_lymphocytes = pd.read_csv("glcm_test2.csv", delimiter=';')
        test_lymphocytes = test_lymphocytes.drop(columns=test_lymphocytes.columns[0])
        test_lymphocytes["class"] = 1
        test_cells = pd.concat([test_blasts, test_lymphocytes])
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

    def NN(self):
        dataset_url = "C:\\Users\\bulig\\.keras\\datasets\\flower_photos"
        data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
        data_dir = pathlib.Path(data_dir)
        image_count = len(list(data_dir.glob('*/*.bmp')))
        print(image_count)
        print(data_dir)

        batch_size = 32
        img_height = 200
        img_width = 200
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        class_names = train_ds.class_names
        np.set_printoptions(edgeitems=10000)
        print(class_names)
        print(train_ds)
        # plt.figure(figsize=(5, 4))
        # for images, labels in train_ds.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(class_names[labels[i]])
        #         plt.axis("off")
        for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
        print(train_ds)
        print(normalization_layer)
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal",
                                                             input_shape=(img_height,
                                                                          img_width,
                                                                          3)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )
        # for images, _ in train_ds.take(1):
        #     for i in range(9):
        #         augmented_images = data_augmentation(images)
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(augmented_images[0].numpy().astype("uint8"))
        #         plt.axis("off")
        #         plt.show()
        # create_model
        num_classes = 2
        model = Sequential([
            data_augmentation,
            layers.experimental.preprocessing.Rescaling(1. / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        # compile
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # description
        model.summary()
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        # education
        epochs = 9
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
        )
        # vizualize_Results
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        print('Точность обучающей',acc)
        print('Точность тестовой', val_acc)
        print('Ошибка обучающей', loss)
        print('Ошибка тестовой', val_loss)




if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

