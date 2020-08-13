import cv2 as cv
import matplotlib.pyplot as plt
from imtools import *

# Детектор углов Харриса

img = cv.imread("../images/skyscraper.webp")
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

h = compute_harris_response(img)
f = get_harris_points(h)

# вывод
plt.figure()
plt.gray()

plt.subplot(111)
plt.imshow(img)
plt.scatter([p[1] for p in f], [p[0] for p in f], s=4, c='g', marker='.')

plt.show()
