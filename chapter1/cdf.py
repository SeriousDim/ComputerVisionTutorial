import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from imtools import *

# Применение выравнивания гистограммы.
# Показ изменения гистограммы и функции cdf
# Функция histeq

img = cv.imread("../images/moscow.jpg")
img = cv.resize(img, (int(img.shape[1]//2), int(img.shape[0]//2)), cv.INTER_CUBIC)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img2, cdf = histeq(img)
x = np.arange(0, 256)

plt.figure(figsize=(7, 7))
plt.gray()

plt.subplot(231)
plt.hist(img.flatten(), 256)

plt.subplot(232)
plt.plot(x, cdf)

plt.subplot(233)
plt.hist(img2.flatten(), 256)

plt.subplot(234)
plt.axis('equal')
plt.axis('off')
plt.imshow(img)

plt.subplot(235)
plt.axis('equal')
plt.axis('off')

plt.subplot(236)
plt.axis('equal')
plt.axis('off')
plt.imshow(img2)

plt.show()
