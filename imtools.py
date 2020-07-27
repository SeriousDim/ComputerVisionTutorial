import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

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
            print("Изображение "+ i +" пропущено: "+e)
            lng -= 1

    img /= lng
    return img.astype(np.uint8)
