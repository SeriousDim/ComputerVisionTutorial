import cv2 as cv
import matplotlib.pyplot as plt
import os
from imtools import *
import homography
import warp

"""
    Составление панорам (не доделано)
"""

# Ишем SIFT-признаки
path = "C:/My folder/Projects/ComputerVision/images/pano/"  # поставить свой путь
imnames = iter(os.listdir(path))
amount = 5

sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

keypoints = {}
points = {}
desc = {}

for i in range(amount):
    ind = next(imnames)
    img = cv.imread(path + ind)
    img = cv.resize(img, (img.shape[1]//3, img.shape[0]//3))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)
    keypoints[i] = kp
    points[i] = []
    for j in range(len(kp)):
        points[i].append(kp[j].pt)
    points[i] = np.array(points[i])
    desc[i] = des

matches = {}

for i in range(amount-1):
    m = bf.match(desc[i], desc[i+1])
    m = sorted(m, key=lambda x: x.distance)
    matches[i] = m

print(len(keypoints[3]), len(points[3]), len(desc[3]), len(matches[0]))

# преобразовать все точки всех совпадений к однородным
def convert_points(img_ind):
    m = matches[img_ind+1]
    p1 = []
    for i in m:
        p1.append(points[img_ind+1][i.queryIdx])
    p1 = np.array(p1).T
    p1 = homography.make_homog(p1)

    p2 = []
    for i in m:
        p2.append(points[img_ind][i.queryIdx])
    p2 = np.array(p2).T
    p2 = homography.make_homog(p2)

    return p1, p2

# Применение RANSAC
model = homography.RansacModel()

fp, tp = convert_points(1)
print(fp, tp)

h_12 = homography.h_from_ransac(fp, tp, model)[0] # изображение 1 к 2

fp, tp = convert_points(0)
h_01 = homography.h_from_ransac(fp, tp, model)[0] # изображение 0 к 1

fp, tp = convert_points(2)
h_32 = homography.h_from_ransac(fp, tp, model)[0] # изображение 3 к 2 (то есть 3 прикладываем слева)

fp, tp = convert_points(3)
h_43 = homography.h_from_ransac(fp, tp, model)[0] # изображение 4 к 3 (то есть 4 прикладываем слева)

# делаем панораму
delta = 2000  # для дополнения и паралелльного переноса
img1 = cv.imread(path + "2.jpg")
img1 = cv.resize(img1, (img1.shape[1]//3, img1.shape[0]//3))
img2 = cv.imread(path + "3.jpg")
img2 = cv.resize(img2, (img2.shape[1]//3, img2.shape[0]//3))
res_12 = warp.panorama(h_12, img1, img2, delta, delta)

img1 = cv.imread(path + "1.jpg")
img1 = cv.resize(img1, (img1.shape[1]//3, img1.shape[0]//3))
res_02 = warp.panorama(np.dot(h_12, h_01), img1, res_12, delta, delta)

img1 = cv.imread(path + "4.jpg")
img1 = cv.resize(img1, (img1.shape[1]//3, img1.shape[0]//3))
res_32 = warp.panorama(h_32, img1, res_02, delta, delta)

img1 = cv.imread(path + "5.jpg")
img1 = cv.resize(img1, (img1.shape[1]//3, img1.shape[0]//3))
res_42 = warp.panorama(np.dot(h_32, h_43), img1, res_32, delta, 2*delta)

cv.imshow(res_42)


