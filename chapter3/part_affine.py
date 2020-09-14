import cv2 as cv
import numpy as np
import scipy.spatial as s
from numpy.linalg import *
from scipy.ndimage import filters as flt
import matplotlib.pyplot as plt
import homography as h

# Применение кусочно-аффиного деформирования

# триангуляция деформируемого изображения
fromimg = cv.imread("../images/leo.jpg")
p = []

x, y = fromimg.shape[1]//4, fromimg.shape[0]//5
for i in range(5):
    for j in range(6):
        p.append([j*y, i*x])

print(p)
p = np.array(p)
tri = h.triangulate_points(p)
p = p.T

# конечное изображение и конечные точки на нем
toimg = cv.imread("../images/figures.jpg")
# выбрать точки вручную с помощью
tp = np.loadtxt("../points.txt")
tp = tp.T

# к однородным координатам
fp = np.vstack((p, np.ones((1, p.shape[1]))))
tp = np.vstack((tp, np.ones((1, tp.shape[1]))))

out = h.pw_affine(fromimg, toimg, fp, tp, tri)

plt.figure()
plt.imshow(fromimg)
plt.scatter(p[:,0], p[:,1], s=10)
plt.show()
