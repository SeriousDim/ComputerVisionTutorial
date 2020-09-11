import cv2 as cv
import numpy as np
import scipy.spatial as s
from numpy.linalg import *
from scipy.ndimage import filters as flt
import matplotlib.pyplot as plt
from homography import *

fromimg = cv.imread("../images/leo.jpg")
p = []

x, y = fromimg.shape[1]//4, fromimg.shape[0]//5
for i in range(4):
    for j in range(5):
        p.append([j*y, i*x])

p = np.array(p)
print(p.T)
tri = triangulate_points(p)

toimg = cv.imread("../images/figures.jpg")
tp = []  # выбрать точки вручную


