import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pydot
import os

# Пример использования PyDot
# Использует пакет pydot, требует приложение Graphviz. Его можно скачать с сайта.
# Устновленный путь нужно записать в PATH

g = pydot.Dot(graph_type='graph')

g.add_node(pydot.Node(str(0), fontcolor="blue"))
for i in range(5):
    g.add_node(pydot.Node(str(i+1)))
    g.add_edge(pydot.Edge(str(0), str(i+1)))
    for j in range(5):
        g.add_node(pydot.Node(str(i+1)+'-'+str(j+1)))
        g.add_edge(pydot.Edge(str(i+1)+'-'+str(j+1), str(i+1)))
g.write_png('graph.png', prog='dot')
