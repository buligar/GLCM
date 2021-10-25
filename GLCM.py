import matplotlib.pyplot as plt
import np as np
import numpy as np

from skimage.feature import greycomatrix, greycoprops
from skimage import io


image = io.imread('Scratch0.jpg') # Загрузка изображения
D=10 # Расстояние смежности
Angles=0
Distances = np.arange(1, D+1, 1)
Angles=[Angles] # Угол
glcm = greycomatrix(image, distances=Distances,
                    angles=Angles,#np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4],
                    levels=256,
                    symmetric=True, normed=True) # Построение МПС

Contrast= greycoprops(glcm, 'contrast') # Текстурный признак Контраст
Dissimilarity=greycoprops(glcm,'dissimilarity') # Текстурный признак несходство
Homogeneity=greycoprops(glcm,'homogeneity')# Текстурный признак Локальная однородность
Asm=greycoprops(glcm,'ASM')# Текстурный признак Угловой второй момент
Energy=greycoprops(glcm,'energy')# Текстурный признак Энергия
Correlation=greycoprops(glcm,'correlation')# Текстурный признак Корреляция


fig = plt.figure() # Для отображения графиков
fig.subplots_adjust(hspace=0.4, wspace=0.4) # Суб-график

plt.subplot(2, 3, 1)
plt.grid(axis = 'both')
plt.title("Контраст")
plt.xticks([i for i in range(0,max(Distances+1))])
plt.plot(Distances,Contrast,marker = 'o')

plt.subplot(2, 3, 2)
plt.grid(axis = 'both')
plt.title("Несходство")
plt.xticks([i for i in range(0,max(Distances+1))])
plt.plot(Distances,Dissimilarity,marker = 'o')

plt.subplot(2, 3, 3)
plt.grid(axis = 'both')
plt.title("Локальная однородность")
plt.xticks([i for i in range(0,max(Distances+1))])
plt.plot(Distances,Homogeneity,marker = 'o')

plt.subplot(2, 3, 4)
plt.grid(axis = 'both')
plt.title("Угловой второй момент")
plt.xticks([i for i in range(0,max(Distances+1))])
plt.plot(Distances,Asm,marker = 'o')

plt.subplot(2, 3, 5)
plt.grid()
plt.title("Энергия")
plt.xticks([i for i in range(0,max(Distances+1))])
plt.plot(Distances,Energy,marker = 'o')

plt.subplot(2, 3, 6)
plt.grid(axis = 'both')
plt.title("Корреляция")
plt.xticks([i for i in range(0,max(Distances+1))])
plt.plot(Distances,Correlation,marker = 'o')

plt.show()
