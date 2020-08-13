import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters as flt

# Наглядное использование производных с помощью Собеля
# и функции (фильтра) Гаусса

img = cv.imread("../images/pca2.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Производные с помощью функции Собеля
sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
sobel = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=5)

sobel2 = np.sqrt(sobel_x**2 + sobel_y**2)

# Производные с помощью фильтра Гаусса
sigma = 1
gx = np.zeros(img.shape)
gy = np.zeros(img.shape)
g = np.zeros(img.shape)
flt.gaussian_filter(img, (sigma, sigma), (0,1), gx)
flt.gaussian_filter(img, (sigma, sigma), (1,0), gy)
flt.gaussian_filter(img, (sigma, sigma), (1,1), g)

g2 = np.sqrt(gx**2 + gy**2)

# Вывод
plt.figure()
plt.gray()

plt.subplot(131)
plt.axis('equal')
plt.axis('off')
plt.imshow(gx)
plt.subplot(132)
plt.axis('equal')
plt.axis('off')
plt.imshow(gy)
plt.subplot(133)
plt.axis('equal')
plt.axis('off')
plt.imshow(g2)

plt.show()

