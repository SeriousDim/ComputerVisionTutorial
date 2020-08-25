import cv2 as cv
import numpy as np
from numpy.linalg import *
from scipy.ndimage import filters as flt
import matplotlib.pyplot as plt

# Реализации алгоритмов из книги

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


def compute_harris_response(img, sigma=3):
    """ Вычислить функцию отклика детектора Харриса для каждого
    пикселя полутонового изображения
    """

    # производные
    imx = np.zeros(img.shape)
    imy = np.zeros(img.shape)
    flt.gaussian_filter(img, (sigma, sigma), (0,1), imx)
    flt.gaussian_filter(img, (sigma, sigma), (1,0), imy)

    # вычислить элементы матрицы Харриса
    wxx = flt.gaussian_filter(imx*imx, sigma)
    wxy = flt.gaussian_filter(imx*imy, sigma)
    wyy = flt.gaussian_filter(imy*imy, sigma)

    # определитель и след матрицы
    w_det = wxx*wxx - wxy**2
    w_trace = wxx + wyy

    return w_det / (w_trace ** 2)


def get_harris_points(harris_img, min_distance=10, threshold=0.3):
    """ Возвращает углы на изображении, построенном по функции отклика Харриса.
    :param min_distance: минимальное число пкс между углами и границей изображения
    """

    # точки-кандидаты
    corner_threshold = harris_img.max() * threshold
    t = (harris_img > corner_threshold) * 1

    # координаты и знаечния кандидатов
    coords = np.array(t.nonzero()).T
    values = [harris_img[c[0],c[1]] for c in coords]

    # отсортировать кандидатов (argsort возвращает индексы)
    index = np.argsort(values)

    # данные о точках-кандидатах в массивах
    allowed_locations = np.zeros(harris_img.shape)
    allowed_locations[min_distance:-min_distance, min_distance:-min_distance] = 1

    # выбрать наилучшие точки с учетом min_distance
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0], coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_distance):(coords[i,0]+min_distance),
            (coords[i,1]-min_distance):(coords[i,1]+min_distance)] = 0

    return filtered_coords


def get_descriptors(img, filtered_coords, width=5):
    """ Для каждой точки вернуть значения пикселей в окрестности этой точки
    шириной 2*wid+1. Предполагается, что выбирались точки с min_distance > min
    """
    desc = []
    for coords in filtered_coords:
        patch = img[coords[0]-width:coords[0]+width+1,coords[1]-width:coords[1]+width+1]
        patch = patch.flatten()
        desc.append(patch)

    return desc


def match(desc1, desc2, threshold=0.5):
    """ Для каждого дескриптора угловой точки в 1-м изображении найти
    соответсвующую ему точку во втором изображении, применяя нормированную
    взаимную корреляцию
    """
    n = len(desc1[0])

    # попарные расстояния
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = ((desc1[i] - np.mean(desc1[i])) / np.std(desc1[i]))
            d2 = ((desc2[j] - np.mean(desc2[j])) / np.std(desc2[j]))
            ncc_value = np.sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value

    ndx = np.argsort(-d)
    scores = ndx[:,0]

    return scores


def match_twosided(desc1, desc2, threshold=0.5):
    """ Двустороний симметричный вариант match()
    """
    match12 = match(desc1, desc2, threshold)
    match21 = match(desc2, desc1, threshold)
    ndx = np.where(match12 >= 0)[0]

    # исключить несимметричные соответсвия
    for n in ndx:
        if match21[match12[n]] != n:
            match12[n] = -1

    return match12


def append_images(img1, img2):
    """ Вернуть изображение, на котором два исходных расположены рядом
    """
    rows1 = img1.shape[0]
    rows2 = img2.shape[0]
    print(rows1, rows2)

    if rows1 < rows2:
        img1 = np.concatenate((img1, np.zeros((rows2-rows1,img1.shape[1]))), axis=0)
    elif rows1 > rows2:
        img2 = np.concatenate((img2, np.zeros((rows1-rows2, img2.shape[1]))), axis=0)

    return np.concatenate((img1, img2), axis=1)


def append_images_3(img1, img2):
    """ Вернуть изображение, на котором два исходных расположены рядом
    """
    rows1 = img1.shape[0]
    rows2 = img2.shape[0]
    print(rows1, rows2)

    if rows1 < rows2:
        img1 = np.concatenate((img1, np.zeros((rows2-rows1,img1.shape[1],img1.shape[2]))), axis=0)
    elif rows1 > rows2:
        img2 = np.concatenate((img2, np.zeros((rows1-rows2, img2.shape[1], img2.shape[2]))), axis=0)

    return np.concatenate((img1, img2), axis=1)


def plot_matches(img1, img2, locs1, locs2, matchscores, show_below=True):
    """ Показать рисунок, на котором соотвественые точки соеденены
    :param locs1, locs2: координаты особых точек
    :param matchscores: результат, возвращенный match()
    :param show_below: показать изображения под картинкой соответсвия
    """
    img3 = append_images(img1, img2)
    if show_below:
        img3 = np.vstack((img3, img3))
    plt.imshow(img3)

    cols1 = img1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            plt.plot([locs1[i][1], locs2[m][1]+cols1], [locs1[i][0], locs2[m][0]], 'c')
    plt.axis('off')

