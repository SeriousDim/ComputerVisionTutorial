import matplotlib.pyplot as plt
import os
from imtools import *

# Применение PCA для датасета MNIST

path = "C:/My folder/mnist/mnist_png/training/1/"  # поставить свой путь
files = iter(os.listdir(path))
amount = 100  # кол-во картинок для чтения

img = cv.imread(path+next(files))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

m, n = img.shape[0:2]  # размер изображения

# линеаризованный массив
mat = []
for j in range(amount):
    bimg = cv.imread(path+next(files))
    bimg = cv.cvtColor(bimg, cv.COLOR_BGR2GRAY)
    bimg = bimg.flatten()
    mat.append(bimg)

mat = np.array(mat)

v, s, mean_m = pca(mat)  # выполнить PCA

# среднее и первые 7 мод
plt.figure()
plt.gray()

plt.subplot(241)
plt.imshow(mean_m.reshape(m, n))

for i in range(7):
    plt.subplot(242+i)
    plt.imshow(v[i].reshape(m, n))

plt.show()
