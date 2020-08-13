import cv2 as cv
import matplotlib.pyplot as plt
from imtools import *

# Соответсвенные точки на двух картинках
# с помощью детектора Харриса

img = cv.imread("../images/town1.jpg")
a = 7
img = cv.resize(img, (img.shape[1]//a, img.shape[0]//a))
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

img2 = cv.imread("../images/town2.jpg")
img2 = cv.resize(img2, (img2.shape[1]//a, img2.shape[0]//a))
img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

wid = 5
harrisim = compute_harris_response(img, 5)
filtered_coords1 = get_harris_points(harrisim, wid+1)
d1 = get_descriptors(img, filtered_coords1, wid)

harrisim2 = compute_harris_response(img2, 5)
filtered_coords2 = get_harris_points(harrisim2, wid+1)
d2 = get_descriptors(img2, filtered_coords2, wid)

matches = match_twosided(d1, d2)
plt.figure()
plt.gray()

plot_matches(img, img2, filtered_coords1, filtered_coords2, matches)

plt.show()
