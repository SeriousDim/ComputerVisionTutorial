import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from pathlib import Path


def compute_rigid_transform(from_points, to_points):
    """ Вычисляет угол поворот, коэффициент мастабирования и
    вектор параллельного переноса для совмещения пар опорных точек.
    В каждом массиве по 3 пары опорных точек.
    from_points - одномерный массив начальных точек в формате: [x1, y1, x2, y2, ...]
    to_points - одномерный массив конечных точек в формате: [x1, y1, x2, y2, ...]
    """

    a = np.array([
        [from_points[0], -from_points[1], 1, 0],
        [from_points[1], from_points[0], 0, 1],
        [from_points[2], -from_points[3], 1, 0],
        [from_points[3], from_points[2], 0, 1],
        [from_points[4], -from_points[5], 1, 0],
        [from_points[5], from_points[4], 0, 1]
    ])

    b = np.array([
        to_points[0],
        to_points[1],
        to_points[2],
        to_points[3],
        to_points[4],
        to_points[5]
    ])

    # метод наименьших квадратов минимизирует норму ||Ax - b||
    # другое определение: минимизирует сумму квадратов разности между b и Ax
    # (Ax - b)^T * (Ax - b) --> min
    a, b, tx, ty = linalg.lstsq(a, b)[0]
    r = np.array([[a, -b], [b, a]]) # матрица поворота с мастабированием

    return r, tx, ty


def rigid_alignment(points, order_list, path):
    """ Собственно сама регистрация изображений.
    Изометрически совмещает изображения (из каталога path) и сохраняет новые
    совмещенные изображения по пути path + "aligned/".
    Вместо функции compute_rigid_transform использованы функции OpenCV
    points - массив точек в формате: [[[x1, y1], [x2, y2], [x3, y3]], ...]
    order_list - список с именами изображений,
    расположенными в нужном для прочтения порядке
    path - каталог с изображениями
    """

    # берем точки из первого изображения в качестве опорных
    ref_points = points[0]
    ref_img = cv.imread(path + order_list[0])

    Path(path + "aligned/").mkdir(parents=True, exist_ok=True)
    # деформация каждого изображения с помощью аффиных преобразований
    for i in range(len(order_list)):
        img = cv.imread(path + order_list[i])
        img = cv.resize(img, (ref_img.shape[1], ref_img.shape[0]))

        p1 = np.float32(points[i])
        p2 = np.float32(ref_points)
        am = cv.getAffineTransform(p1, p2)
        img_a = cv.warpAffine(img, am, (img.shape[1], img.shape[0]))

        cv.imwrite(path + "aligned/" + order_list[i], img_a)

