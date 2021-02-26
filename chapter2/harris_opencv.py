import cv2 as cv
import matplotlib.pyplot as plt
from imtools import *

"""
    Детектор углов Харриса от OpenCV
"""

img = cv.imread("../images/chess.jpg")
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

dst = cv.cornerHarris(gray, 2, 3, 0.04)

dst_norm = np.empty(dst.shape, dtype=np.float32)
cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
dst_norm_scaled = cv.convertScaleAbs(dst_norm)

for i in range(dst_norm.shape[0]):
    for j in range(dst_norm.shape[1]):
        if int(dst_norm_scaled[i,j]) > 110:
            img = cv.circle(img, (j, i), 3, (0, 255, 0), 1)

# вывод
plt.figure()
plt.gray()

plt.subplot(111)
plt.imshow(img)

plt.show()
