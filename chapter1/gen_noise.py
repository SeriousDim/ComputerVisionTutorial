import cv2 as cv
import matplotlib.pyplot as plt
from numpy import random

# Генерация шума

img = cv.imread("../images/moscow.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

rnd = 50 * random.standard_normal(img.shape)
noise = img + rnd

den = cv.fastNlMeansDenoising(img, None, 10, 7 ,21)

plt.figure()
plt.gray()

plt.imshow(noise)

plt.show()
