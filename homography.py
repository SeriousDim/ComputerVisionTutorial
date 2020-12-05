import cv2 as cv
import numpy as np
import scipy.spatial as s
from numpy.linalg import *
from scipy.ndimage import filters as flt
import matplotlib.pyplot as plt

def normalize(points):
    """ Нормировать коллецию точек в однородных координатах так,
    чтобы последняя строка была равна 1
    """
    for row in points:
        row /= points[-1]
    return points


def make_homog(points):
    """ Преобразовать множество точек (массив dim*n) в однородные координаты
    """
    return np.vstack((points, np.ones((1, points.shape[1]))))


def h_from_points(fp, tp):
    """ Найти гомографию H, отображающую fp в tp, методом DLT
    (прямого линейного преобразования). Хорошая обусловленность точек обеспечивается
    автоматически
    """
    if fp.shape != tp.shape:
        raise RuntimeError("Размеры fp и tp не совпадают")

    # обеспечиваем хорошую обусловленность точек (важно для вычислений):
    # нормируем так, чтобы среднее стало = 0, а отклонение = 1
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    c1 = np.diag([1/maxstd, 1/maxstd, 1])
    c1[0][2] = -m[0]/maxstd
    c1[1][2] = -m[1]/maxstd
    fp = np.dot(c1, fp)

    m = np.mean(tp[:2], axis=1)
    maxstd = np.max(np.std(tp[:2], axis=1)) + 1e-9
    c2 = np.diag([1 / maxstd, 1 / maxstd, 1])
    c2[0][2] = -m[0] / maxstd
    c2[1][2] = -m[1] / maxstd
    tp = np.dot(c2, tp)

    # создать матрицу A для метода DLT (A*h=0), по две строки для каждой
    # пары соответсвенных точек
    nbr = fp.shape[1]
    a = np.zeros((2*nbr, 9))
    for i in range(nbr):
        a[2*i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i]*fp[0][i], tp[0][i]*fp[1][i], tp[0][i]]
        a[2*i+1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i]*fp[0][i], tp[1][i]*fp[1][i], tp[1][i]]

    u, s, v = np.linalg.svd(a)
    h = v[8].reshape((3,3))

    # обратное преобразование, компенсирующее обусловливание
    h = np.dot(np.linalg.inv(c2), np.dot(h, c1))

    return h / h[2,2] # нормировка


def triangulate_points(p):
    """ Триангуляция точек по Делоне
    p = np.array([[x1, y1], ...])
    """
    tri = s.Delaunay(p)
    return tri.simplices


def alpha_for_triangle(points, m, n):
    """ Создает альфа-отображение размера (m, n) для
    треугольника с вершинами points

    Формат points: np.int32([[[x1, y1], [x2, y2], [x3, y3]]]) (см. док. cv.fillPoly)
    """
    alpha = np.zeros((m, n))
    alpha = cv.fillPoly(alpha, points, 1)
    return alpha


def pw_affine(fromimg, toimg, fp, tp, tri):
    """ Деформирвоать треугольные блоки изображения.
    Вместо функции h_from_points используются функции OpenCV
    fromimg - деформируемое изорбажение
    toimg - конечное изображение
    fp - исходные точки в однородных координатах (на деформ. изобр.)
    tp - конечные точки в однородных координатах (на конечном изобр.)
    tri - триангуляция
    """

    img = toimg.copy()
    is_color = len(fromimg.shape) == 3  # цветное ли изображение
    im_t = np.zeros(img.shape, np.uint8)  # изобр., на которое будет переносится исходное

    for t in tri:
        # аффинное преобразвание
        i, j, k = t[0], t[1], t[2]
        p1 = np.float32([fp[:2,i], fp[:2,j], fp[:2,k]])
        p2 = np.float32([tp[:2,i], tp[:2,j], tp[:2,k]])
        h = cv.getAffineTransform(p1, p2)

        if is_color:
            for clr in range(fromimg.shape[2]):
                im_t[:,:,clr] = cv.warpAffine(fromimg[:,:,clr], h, (img.shape[1], img.shape[0]))
        else:
            im_t = cv.warpAffine(fromimg, h, (img.shape[1], img.shape[0]))

        # альфа-отображение для треугольника
        alpha = alpha_for_triangle(np.int32([p2]), img.shape[0], img.shape[1])

        # добавить треугольник в изображение
        img[alpha > 0] = im_t[alpha > 0]

    return img




