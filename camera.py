import scipy.linalg as la
import numpy as np

class Camera:
    """
        Класс для представления камеры с точечной диафрагмой
    """

    def __init__(self, p):
        self.p = p  # матрциа камеры P = K[R|t]
        self.k = None  # калибрововчная матрица
        self.r = None  # матрица поворота
        self.t = None  # паралелльный перенос
        self.c = None  # центр камеры

    def project(self, x):
        """
            Спроецировать точки из массива X (4хN) и нормировать координаты
            :param x: массив 4хN
        """

        x = np.dot(self.p, x)
        for i in range(3):
            x[i] /= x[2]
        return x

def rotation_matrix(a):
    """
        Матрица поворота вокруг оси вектора A в трехмерном пространстве
        :param a: вектор, вокргу которого идет поворот
    """

    buf = np.eye(4)
    buf[:3, :3] = la.expm()


