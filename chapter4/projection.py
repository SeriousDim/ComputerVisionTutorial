import camera
import numpy as np
import matplotlib.pyplot as plt

"""
    Используем класс Camera для проекции точек трехмерной модели
    на камеру с точечной диафрагмой
"""

# автор предлагает загрузить файл с моделью, однако он недоступен
# файл p3d содержит просто numpy-подобный массив
points = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
          [0, 0, 2], [2, 0, 2], [0, 2, 2], [2, 2, 2]]  # куб
points = np.array(points).T
points = np.vstack((points, np.ones(points.shape[1])))  # однородные координаты в 3D пространстве

# настроить камеру
k = np.eye(3)
t = np.array([[7], [5], [-10]])
p = np.hstack((k, t))

cam = camera.Camera(p)
x = cam.project(points)

# показать проекцию
plt.figure()
plt.plot(x[0], x[1], 'k.')
plt.show()


#

