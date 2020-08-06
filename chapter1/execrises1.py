import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import filters as flt
from scipy.ndimage import measurements, morphology
from imtools import *

from imtools import *

plt.figure()
plt.gray()

# 2. Нерезкое маскирование
img = cv.imread("../images/moscow.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
bl = cv.GaussianBlur(gray, (9,9), 10)
k = 0.5
total = cv.addWeighted(gray, 1+k, bl, -k, 0)  # изображение с увеличенной резкостью

# 6. Размеры объктов
img = cv.imread("../images/asteroids.png", cv.IMREAD_UNCHANGED)
b, g, r, a = cv.split(img)
ret, bin = cv.threshold(a, 128, 255, cv.THRESH_BINARY)
labels, num = measurements.label(bin)
h, bins = np.histogram(labels.flatten(), 256, density=True)

# 7. Центры масс
img = cv.imread("../images/asteroids.png", cv.IMREAD_UNCHANGED)
b, g, r, a = cv.split(img)
ret, bin = cv.threshold(a, 128, 255, cv.THRESH_BINARY)
labels, num = measurements.label(bin)

cnt = measurements.center_of_mass(bin, labels, [i for i in range(1, num+1)])
print(cnt)
for c in cnt:
    img = cv.circle(img, (int(c[1]), int(c[0])), 4, (0, 255, 0, 255), 3)

# вывод
plt.subplot(221)
plt.imshow(labels)

plt.subplot(222)
plt.bar(bins[1:-1], h[1:])

plt.subplot(223)
plt.imshow(img)

plt.subplot(224)
plt.hist(labels.flatten(), 256, width=1)

plt.show()