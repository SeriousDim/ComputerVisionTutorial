import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import image_registration as reg

""" Регистрация изображений
"""

data = []
file_names = []
path_txt = "C:/My folder/Projects/ComputerVision/txt/"
path_img = "C:/My folder/Projects/ComputerVision/images/dima/resised/"

def read_data():
    """ Чтение координат на точках и имен файлов.
    Точки ставятся с помощью файла select_points.py
    """
    with open(path_txt + "dima_points.pkl", "rb") as f:
        global data
        data = pickle.load(f)

    global file_names
    with open(path_txt + "dima_points_order.txt", "r") as f:
        file_names = f.readlines()

    file_names = [s[:-1] for s in file_names]  # удалить '\n' из конца строк

def show_points(index):
    """ Показать точки на изображении
    """

    img = cv.imread(path_img + file_names[index])
    plt.figure()
    plt.imshow(img)

    plt.scatter(data[index][0][0], data[index][0][1], c='r')
    plt.scatter(data[index][1][0], data[index][1][1], c='g')
    plt.scatter(data[index][2][0], data[index][2][1], c='b')

    plt.show()

read_data()
#reg.rigid_alignment(data, file_names, path_img)

print(data)
print(file_names)

show_points(0)
