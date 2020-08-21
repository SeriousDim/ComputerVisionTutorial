import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pydot
import os

# Визуализация связанных изображений
# Читает файл match.txt, созданный программой corresponding_many_photos.py
# Использует пакет pydot, требует приложение Graphviz. Его можно скачать с сайта.
# Устновленный путь нужно записать в PATH
match_scores = np.loadtxt("../match.txt")
g = pydot.Dot(graph_type='graph')

path = "../images/redsquare/"
files = os.listdir(path)
num = len(files)

threshold = 800

for i in range(num):
    for j in range(i+1, num):
        if match_scores[i,j] > threshold:
            # первое изображение в паре
            img = cv.imread(path+files[i])
            img = cv.resize(img, (200, int(200/img.shape[1]*img.shape[0])), cv.INTER_CUBIC)
            f = str(i)+'-mini.png'
            cv.imwrite("../images/redsquare_mini/"+f, img)
            g.add_node(pydot.Node(str(i), fontcolor='transparent', shape='rectangle', image="C:/My folder/Projects/ComputerVision/images/redsquare_mini/"+f))

            # второе изображение в паре
            img = cv.imread(path+files[j])
            img = cv.resize(img, (100, int(100 / img.shape[1]*img.shape[0])), cv.INTER_CUBIC)
            f = str(j) + '-mini.png'
            cv.imwrite("../images/redsquare_mini/" + f, img)
            g.add_node(pydot.Node(str(j), fontcolor='transparent', shape='rectangle', image="C:/My folder/Projects/ComputerVision/images/redsquare_mini/"+f))

            g.add_edge(pydot.Edge(str(i), str(j)))

g.write_png('redsquare_graph.png')
