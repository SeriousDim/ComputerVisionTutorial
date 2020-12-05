import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt

# Более полезное применение PCA
# Read the OpenCV docs: https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    # угол данной оси (и вектора pq) относительно Ox
    angle = atan2(p[1] - q[1], p[0] - q[0])  # перемещаем вектор в начало координат и ищем угол
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0])) # длина вектора pq

    # увеличиваем длину вектора в scale раз
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)

    cv.line(img, (int(p[0]), int(p[1])), ((int(q[0])), int(q[1])), color, 1, cv.LINE_AA)


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i,0,0]
        data_pts[i, 1] = pts[i,0,1]

    mean = np.empty((0))
    mean, eigenvectors, eigenvals = cv.PCACompute2(data_pts, mean)

    center = (int(mean[0,0]), int(mean[0,1]))

    cv.circle(img, center, 3, (255, 0, 255), 2)
    p1 = (center[0] + 0.02 * eigenvectors[0,0] * eigenvals[0,0], center[1] + 0.02 * eigenvectors[0,1] * eigenvals[0,0])
    p2 = (center[0] - 0.02 * eigenvectors[1,0] * eigenvals[1,0], center[1] - 0.02 * eigenvectors[1,1] * eigenvals[1,0])
    drawAxis(img, center, p1, (0, 255, 0), 1)
    drawAxis(img, center, p2, (255, 255, 0), 5)

    print(eigenvectors)

    # угол между осью Ox и вектором eigenvector[0]
    # atan2(y, x) - угол между осью Ox и вектором V(x, y)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])

    return angle


src = cv.imread("../images/leo.jpg")
cv.imshow('src', src)

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

_, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

contours, h = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for i, c in enumerate(contours):
    area = cv.contourArea(c)
    if area < 100 or area > 100000:
        continue

    cv.drawContours(src, contours, i, (0, 0, 255), 2)
    getOrientation(c, src)

cv.imshow('out', src)
cv.waitKey(0)
