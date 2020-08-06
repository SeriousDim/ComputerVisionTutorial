import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements, morphology

# Статья на Хабре о математической морфологии: https://habr.com/ru/post/113626/

img = cv.imread("../images/asteroids2.png", cv.IMREAD_UNCHANGED)
b, g, r, a = cv.split(img)
ret, bin = cv.threshold(a, 128, 255, cv.THRESH_BINARY)

# одна из операций морфологии: размыкание (opening)
# размыкание = эрозия, затем сразу же наращивание
bin_open = morphology.binary_opening(bin, np.ones((20,5)), iterations=2)

labels, num = measurements.label(bin_open)
print(num)

plt.figure()

plt.subplot(131)
plt.imshow(bin)

plt.subplot(132)
plt.imshow(bin_open)

plt.subplot(133)
plt.imshow(labels)

plt.show()
