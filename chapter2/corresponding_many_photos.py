import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters as flt
import os

"""
    Загружаем фото Красной площади, выделяем в них SIFT-признаки,
    и сравниваем друг с другом. Так находим похожие фото.
"""

path = "../images/redsquare/"
files = os.listdir(path)
num = len(files)

sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
# матрица связи, чем больше match_scores[i,j], тем больше похожи
# изображения i и j
match_scores = np.zeros((num, num))

for i in range(num):
    print("Comparing with image "+files[i])
    img = cv.imread(path + files[i])
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (int(img.shape[1]//2), int(img.shape[0]//2)), cv.INTER_CUBIC)
    kp, des = sift.detectAndCompute(img, None)
    for j in range(i, num): # берем только верхний треугольник
        cimg = cv.imread(path + files[j])
        cimg = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
        cimg = cv.resize(cimg, (int(cimg.shape[1] // 2), int(cimg.shape[0] // 2)), cv.INTER_CUBIC)
        kp2, des2 = sift.detectAndCompute(cimg, None)
        matches = bf.match(des, des2)
        match_scores[i,j] = len(matches)

np.savetxt("../match.txt", match_scores, fmt="%.0d")
