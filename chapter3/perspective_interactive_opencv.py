import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# Need repair

img = cv.imread("../images/rent.jpg")
out = img.copy()

x = []
y = []

fig, ax = plt.subplots()
line, = ax.plot(x, y, marker='o')

def onpick(event):
    m_x, m_y = event.x, event.y
    px, py = ax.transData.inverted().transform([m_x, m_y])
    x.append(px)
    y.append(py)
    line.set_xdata(x)
    line.set_ydata(y)
    if len(x) == 4:
        pt = [[x[i], y[i]] for i in range(4)]
        pout = [[0 ,500], [500, 500], [500, 0], [0, 0]]
        m = cv.getPerspectiveTransform(pt, pout)
        out = cv.warpPerspective(img, m, (500, 500))
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onpick)

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(out)

plt.show()
