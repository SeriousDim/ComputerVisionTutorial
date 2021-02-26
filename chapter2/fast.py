import cv2 as cv
import matplotlib.pyplot as plt
from imtools import *

"""
    Docs: https://docs.opencv.org/master/df/d0c/tutorial_py_fast.html
    В документации - описание алгоритма FAST и его использование с
    деревьями решений (раздел Machine Learning a Corner Detector)
"""

img = cv.imread("../images/chess.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
fast = cv.FastFeatureDetector_create(threshold=40, type=2)

plt.figure()

kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, (0, 255, 0))
plt.subplot(121)
plt.imshow(img2)

# Стандартные параметры
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )


# Убрать nonmaxSuppression. NonmaxSuppression исключает нахождение нескольких в
# одной какой-то области
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
img3 = cv.drawKeypoints(img, kp, None, color=(0, 255,0))
plt.subplot(122)
plt.imshow(img3)
print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))

plt.show()
