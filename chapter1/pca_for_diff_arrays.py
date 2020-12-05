import matplotlib.pyplot as plt
import os
from imtools import *
import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt

a11 = [1, 3, 5, 7, 9, 11, 13, 15]
a12 = [15, 82, 45, 91, 22, 26, 34, 61]

ac = np.array([a11, a12], np.float32)

#v, s, m = pca(a11)
m2, eigenvec, eigenvals = cv.PCACompute2(ac, np.empty((0)))

print("===========")
print(eigenvals)
