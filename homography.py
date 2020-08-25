import cv2 as cv
import numpy as np
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

