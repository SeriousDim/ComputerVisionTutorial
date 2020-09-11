import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

img = cv.imread("../images/chess.jpg")

# Деформация изображений
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

# scaling
img2 = cv.resize(img, (img.shape[1]//2, img.shape[0]//2), cv.INTER_CUBIC)
img3 = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

# translation
m = np.float32([[1, 0, 100], [0, 1, 50]]) # перемещение на (100, 50)
img_t = cv.warpAffine(img, m ,(img.shape[1], img.shape[0]))

# rotation
rm = cv.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 60, 1)
img_r = cv.warpAffine(img, rm, (img.shape[1], img.shape[0]))

# affine
p1 = np.float32([[533, 10], [1267, 194], [978, 944]]) # 3 точки
p2 = np.float32([[300, 100], [800, 100], [800, 600]])
am = cv.getAffineTransform(p1, p2)
img_a = cv.warpAffine(img, am, (img.shape[1], img.shape[0]))

# perspective
p1 = np.float32([[533, 10], [1267, 194], [978, 944], [10, 497]]) # 4 точки
p2 = np.float32([[0, 800], [800, 800], [800, 0], [0, 0]])
pm = cv.getPerspectiveTransform(p1, p2)
img_p = cv.warpPerspective(img, pm, (800, 800))

# вывод
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(img_p)
plt.show()
