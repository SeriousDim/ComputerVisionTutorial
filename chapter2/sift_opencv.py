import cv2 as cv
import matplotlib.pyplot as plt
from imtools import *

"""
    Использование SIFT от OpenCV. Поиск точек, дескрипторов и
    сопоставление точек на двух картинках, выбор одной точки.
    НЕОБХОДИМО для работы установить OpenCV из исходников
    по инструкции. Инстуркция в файле OpenCV from sources.md в корне проекта
    Туториал от OpenCV: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
    
    Объяснение представления ключевых точек и дескрипторов в OpenCV здесь:
    https://ianlondon.github.io/blog/how-to-sift-opencv/
"""

img1 = cv.imread("../images/town1.jpg")
img2 = cv.imread("../images/town2.jpg")
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

gray1 = cv.addWeighted(gray1, 1, gray1, 0, -99)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print(kp1[100].pt)

out1 = np.zeros(gray1.shape)
out1 = cv.drawKeypoints(gray1, kp1, out1, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

out2 = np.zeros(gray2.shape)
out2 = cv.drawKeypoints(gray2, kp2, out1, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

#print(kp1[5442].angle)
#print(des1[5442])

# Feature matching with OpenCV
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x: x.distance)
output = cv.drawMatches(gray1, kp1, gray2, kp2, matches[:50], img2.copy(), flags=0)

print(matches[555].distance) # trainIdx, queryIdx

# Get some matched point
i1 = matches[555].queryIdx
i2 = matches[555].trainIdx
p1 = kp1[i1].pt
p2 = kp2[i2].pt
p1 = (int(p1[0]), int(p1[1]))
p2 = (int(p2[0]), int(p2[1]))
img1 = cv.circle(gray1, p1, 30, (0, 255, 0), 3)
img2 = cv.circle(gray2, p2, 30, (0, 255, 0), 3)

print(i1, i2)

plt.figure()
plt.gray()

plt.imshow(output)

plt.show()