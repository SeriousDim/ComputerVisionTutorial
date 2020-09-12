import cv2 as cv
import matplotlib.pyplot as plt

""" Просто выводит изображение
"""

img = cv.imread("images/figures.jpg");

plt.figure()
plt.imshow(img)
plt.show()
