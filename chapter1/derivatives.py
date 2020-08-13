import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters as flt

# Получение производных изображения с помощью функций OpenCV и scipy

img = cv.imread("../images/leo.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Производные с помощью функции Собеля
sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
sobel = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=5)

# Производные с помощтю функции Шарра
sh_x = cv.Scharr(img, cv.CV_64F, 1, 0)
sh_y = cv.Scharr(img, cv.CV_64F, 0, 1)

# Производные с помощью функции Лапласа
pre = cv.GaussianBlur(img, (3,3), 0)
lap = cv.Laplacian(pre, cv.CV_64F, ksize=9)

# Производные с помощью фильтра Гаусса
gx = np.zeros(img.shape)
gy = np.zeros(img.shape)
g = np.zeros(img.shape)
flt.gaussian_filter(img, (5,5), (0,1), gx)
flt.gaussian_filter(img, (5,5), (1,0), gy)
flt.gaussian_filter(img, (5,5), (1,1), g)

# Вывод
plt.figure()
plt.gray()

plt.subplot(241)
plt.axis('equal')
plt.axis('off')
plt.imshow(sobel_x)
plt.subplot(242)
plt.axis('equal')
plt.axis('off')
plt.imshow(sobel_y)
plt.subplot(243)
plt.axis('equal')
plt.axis('off')
plt.imshow(sobel)

plt.subplot(244)
plt.axis('equal')
plt.axis('off')
plt.imshow(sh_x)
plt.subplot(245)
plt.axis('equal')
plt.axis('off')
plt.imshow(sh_y)

plt.subplot(246)
plt.axis('equal')
plt.axis('off')
plt.imshow(gx)
plt.subplot(247)
plt.axis('equal')
plt.axis('off')
plt.imshow(gy)
plt.subplot(248)
plt.axis('equal')
plt.axis('off')
plt.imshow(g)

cv.imshow("original", img)
cv.imshow("laplacian", lap)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
