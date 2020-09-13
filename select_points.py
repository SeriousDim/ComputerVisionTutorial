import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

""" Выбирает точки по нажатию
"""

img = cv.imread("images/figures.jpg");

plt.figure()
plt.imshow(img)

points = plt.ginput(30, 10000)
np.savetxt("points.txt", points)
print(len(points))

plt.show()
