import matplotlib.pyplot as plt
from imtools import *

# Наглядный результат выравнивание гистограммы.
# Результат - нормировка яркости изображения

img = cv.imread("../images/moscow.jpg")
img = cv.resize(img, (int(img.shape[1]//2), int(img.shape[0]//2)), cv.INTER_CUBIC)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img2, cdf = histeq(img)
img2 = img2.astype(np.uint8)

plt.imshow(img2)

cv.imshow("before", img)
cv.imshow("after", img2)

plt.show()

cv.waitKey(0)