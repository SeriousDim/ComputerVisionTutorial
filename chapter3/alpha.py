import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import time

""" Альфа-отображение (маска) создается с помощью функции fillPoly
"""

img = np.zeros((720*2, 1280*2))
img = cv.fillPoly(img, np.array([[[0, 0], [0, 720*2], [1280*2, 720*2]]]), 1)

plt.imshow(img)
plt.show()
