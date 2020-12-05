import matplotlib.pyplot as plt
import os
from imtools import *
import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt

""" Сравнение резульатов PCA, подсчитанных с помошью SVD и функции OpenCV
"""

path = "C:/My folder/Projects/ComputerVision/images/dima/resised/aligned/"  # поставить свой путь
files = iter(os.listdir(path))
amount = 30  # кол-во картинок для чтения

mat = []
for j in range(amount):
    bimg = cv.imread(path+next(files))
    bimg = cv.cvtColor(bimg, cv.COLOR_BGR2GRAY)
    bimg = bimg.flatten()
    mat.append(bimg)

mat = np.array(mat)
mean = np.empty((0))

v1, s1, mean1 = pca(mat)
mean2, eigenvec, eigenvals = cv.PCACompute2(mat, mean)

print(v1)
print(s1)

print("=============")

print(eigenvec)
print(eigenvals)
