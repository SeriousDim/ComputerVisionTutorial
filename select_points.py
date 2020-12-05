import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

""" Выбирает точки по нажатию6 для одной или нескольких изображений
    с помощью метода ginput.
    Если фотографий несколько: то, в каком порядке изображения прочитались,
    сохраняется в файл .txt
    Сами точки сохраняются в виде массива в фaйл .pkl с помощью Pickle
"""

def select_points(img, points_num, timeout=10000):
    img = cv.imread(img)

    plt.figure()
    plt.imshow(img)

    points = plt.ginput(points_num, timeout)  # кол-во точек и таймаут(сек)
    return points

def select_points_multiple(path, img_list, points_num, timeout=10000):
    points_list = []
    for name in img_list:
        img = cv.imread(path + name)

        plt.figure()
        plt.imshow(img)
        points = plt.ginput(points_num, timeout)
        points_list.append(points)

        plt.close()
    return points_list

path = "C:/My folder/Projects/ComputerVision/images/dima/resised/"
img_list = os.listdir(path)
print(img_list)

with open("C:/My folder/Projects/ComputerVision/txt/dima_points_order.txt", "w") as f:
    for i in img_list:
        f.write(i + "\n")


points_list = select_points_multiple(path, img_list, 3)  # левый глаз, правый глаз, рот
print(points_list)
with open("C:/My folder/Projects/ComputerVision/txt/dima_points.pkl", "wb") as f:
    pickle.dump(points_list, f)