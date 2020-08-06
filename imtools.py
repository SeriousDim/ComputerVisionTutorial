import cv2 as cv
import numpy as np
from numpy.linalg import *


def histeq(img, nbr_bins=256):
    """ Выравнивание гистограммы изображения в оттенках серого """
    h, bins = np.histogram(img.flatten(), nbr_bins, density=True)  # гистограмма
    cdf = h.cumsum()  # функция распределения(cdf); здесь - накапливаемая сумма
    cdf = 255 * cdf / cdf[-1]  # нормируем до [0, 255]

    # присваиваем новые значения изображения в соответсвии функции cdf
    # новые_значения = np.interp(исходные_значения, x-координаты cdf, y-координаты cdf)
    img2 = np.interp(img.flatten(), bins[:-1], cdf)
    return img2.reshape(img.shape), cdf


def compute_average(img_list):
    """ Вычислить среднее списка изображений """
    img = cv.imread(img_list[0])
    img = img.astype(np.float32)
    lng = len(img_list)

    for i in img_list[1:]:
        try:
            ni = cv.imread(img_list[i])
            ni = img.astype(np.float32)
            img += ni
        except Exception as e:
            print("Изображение " + i + " пропущено: " + e)
            lng -= 1

    img /= lng
    return img.astype(np.uint8)


def pca(x):
    """ Метод главных компонент (PCA)
    вход: матрица Х, в которой обучающие данные хранятся в виде линеаризованных массивов,
    по одному в каждой строке
    выход: матрица проекции (наиболее важные измерения в начале),
    дисперсия (отклонение) и среднее
    """
    num_data, dim = x.shape  # кол-во измерений
    # центрирование данных
    mean_x = np.mean(x, axis=0)
    x = x - mean_x

    if dim > num_data:
        # PCA с компактным трюком
        m = np.dot(x, x.T)
        e, ev = np.linalg.eigh(m)  # собственные значения и векторы
        tmp = np.dot(x.T, ev).T
        v = tmp[::-1]  # берем последние собственные векторы
        s = np.sqrt(e)[::-1]  # собственные значения в порядке убывания
        for i in range(v.shape[1]):
            v[:,i] /= s
    else:
        # PCA с сингулярным разложением (SVD)
        u, s, v = np.linalg.svd(x)
        v = v[:num_data]  # возвращаем только первые num_data строк

    return v, s, mean_x


def chambolle_denoise(img, u_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """ Решатель модели ROF (Рудина-Ошера-Фатеми) на базе алгоритма
    Шамболя. Используется для очистки изображения от шума

    tolerance - вес члена, регуляризирующего TV (total variation - полная вариация,
    сумма норм градиентов)
    tau - величина шага
    tv_weight - допуск в условии остановки
    """
    m, n = img.shape

    # инициализация
    u = u_init
    px = img  # компонента х двойственной задачи
    py = img  # компонента y двойственной задачи
    error = 1

    while (error > tolerance):
        u_old = u
        # градиент переменной прямой задачи
        grad_ux = np.roll(u, -1, axis=1)-u  # компонента х градиента U
        grad_uy = np.roll(u, -1, axis=0)-u  # компонента y градиента U

        # изменить переменную двойственной задачи
        px_new = px + (tau/tv_weight)*grad_ux
        py_new = py + (tau/tv_weight)*grad_uy
        norn_new = np.maximum(1, np.sqrt(px_new**2+py_new**2))
        px = px_new/norn_new
        py = py_new/norn_new

        # изменить переменную прямой задачи
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)

        divp = (px - rx) + (py - ry)  # дивергенция двойственного поля
        u = img + tv_weight*divp

        # пересчитать погрешность
        error = norm(u-u_old) / np.sqrt(n*m)

    return u, img - u  # очищенное от шумов изображение и остаточная текстура

