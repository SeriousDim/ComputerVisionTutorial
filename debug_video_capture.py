import cv2 as cv
import matplotlib.pyplot as plt

""" Просто выводит изображение с веб-камеры
"""

cam = cv.VideoCapture(0)

_, f = cam.read()
cv.imshow('cam', f)

cv.waitKey(20)
