import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../images/leo.jpg")
#img = cv.resize(img, (img.shape[1]//2, img.shape[0]//2))

cv.imshow('original', img)
plt.figure()

# 2D Convolution
kernel = np.ones((5, 5), np.float32)/25
flt = cv.filter2D(img, -1, kernel)
# cv.imshow('filtered 1', flt)

# Averaging
blur = cv.blur(img, (5,5))
# cv.imshow('blur 1', blur)

# gaussian
g = cv.GaussianBlur(img, (5,5), 0)

#cv.imshow('gaussian', g)

# median
med = cv.medianBlur(img, 5)

# cv.imshow('median', med)

# bilateral
bil = cv.bilateralFilter(img, 9, 300, 300)

cv.imshow('bilateral', bil)

cv.waitKey(0)
cv.destroyAllWindows()
