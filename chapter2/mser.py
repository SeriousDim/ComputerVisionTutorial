import cv2 as cv
import matplotlib.pyplot as plt
from imtools import *

#Create MSER object
mser = cv.MSER_create()

#Your image path i-e receipt path
img = cv.imread('../images/chess.jpg')

#Convert to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

vis = img.copy()

#detect regions in gray scale image
regions, _ = mser.detectRegions(gray)

hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv.polylines(vis, hulls, 1, (0, 255, 0))

cv.imshow('img', vis)

cv.waitKey(0)
