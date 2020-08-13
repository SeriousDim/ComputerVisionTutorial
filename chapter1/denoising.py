import cv2 as cv
import matplotlib.pyplot as plt
from imtools import *

# Избавление изображения от шума с помощью разных алгоритмов
# Chambolle ROF (из книги, медленее)
# и Non-local Mean Denoising (OpenCV, быстрее)

img = cv.imread("../images/noise1.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Избавление от шума с помощью ROF Chambolle: только grayscale
#r1, t1 = chambolle_denoise(gray, gray)

# C помощью функций OpenCV
r2 = cv.fastNlMeansDenoising(img, None, 10, 15, 29)

plt.figure()
plt.gray()

#plt.subplot(221)
#plt.imshow(img)

#plt.subplot(222)
#plt.imshow(r1)

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(r2)

plt.show()
