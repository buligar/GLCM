import matplotlib.pyplot as plt
import np as np

from skimage.feature import greycomatrix, greycoprops
from skimage import io
import sys
import numpy as np

from PyQt5 import QtCore, QtWidgets, uic

import matplotlib
matplotlib.use('QT5Agg')

import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('GLCM.ui', self)
        self.fig = plt.figure()  # Для отображения графиков
        self.fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Суб-график
        self.plotWidget = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self.GraphWidget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)
        self.addFunctions() #вызов функций кнопок

    def addFunctions(self):
        self.pushButton.clicked.connect(lambda: self.load())

    def load(self):
        image = io.imread('Scratch0.jpg')  # Загрузка изображения
        self.value = self.spinBox.value() # Расстояние смежности
        Angles = 0
        Distances = np.arange(1, self.value + 1, 1)
        Angles = [Angles]  # Угол
        glcm = greycomatrix(image, distances=Distances,
                            angles=Angles,  # np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4],
                            levels=256,
                            symmetric=True, normed=True)  # Построение МПС

        Contrast = greycoprops(glcm, 'contrast')  # Текстурный признак Контраст
        Dissimilarity = greycoprops(glcm, 'dissimilarity')  # Текстурный признак несходство
        Homogeneity = greycoprops(glcm, 'homogeneity')  # Текстурный признак Локальная однородность
        Asm = greycoprops(glcm, 'ASM')  # Текстурный признак Угловой второй момент
        Energy = greycoprops(glcm, 'energy')  # Текстурный признак Энергия
        Correlation = greycoprops(glcm, 'correlation')  # Текстурный признак Корреляция



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

        plt.show()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())