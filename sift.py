import cv2 as cv
import os
import numpy as np
import scipy.spatial as s
from numpy.linalg import *
from scipy.ndimage import filters as flt
import matplotlib.pyplot as plt
import ransac

"""
    Модуль работы с приложением для поиска SIFT-признаков
    
"""

def process_image(img_name, result_name, params="--edge-thresh 10 --peak-thresh 5"):
    """
        Находит SIFT-признаки и сохраняет их в файл
        :param img_name:
        :param result_name:
        :param params: параметры для программы
    """
    if img_name[-3:] != 'pgm':
        # создать pgm-файл
        img = cv.imread(img_name)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imwrite("C:/My folder/Projects/ComputerVision/images/sift/tmp.pgm", img)

        cmd = str("sift "+img_name+" --output="+result_name+" "+params)
        os.system(cmd)

def read_features(filename):
    f = np.loadtxt(filename)
    return f[:, :4], f[:, 4:]  # положения признаков, дескрипторы
