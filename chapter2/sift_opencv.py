import cv2 as cv
import matplotlib.pyplot as plt
from imtools import *

# Объяснение представления ключевых точек и дескрипторов здесь:
# https://ianlondon.github.io/blog/how-to-sift-opencv/

img1 = cv.imread("../images/town1.jpg")
img2 = cv.imread("../images/town2.jpg")
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print(len(kp1))

out1 = np.zeros(gray1.shape)
out1 = cv.drawKeypoints(gray1, kp1, out1, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

out2 = np.zeros(gray2.shape)
out2 = cv.drawKeypoints(gray2, kp2, out1, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

#print(kp1[5442].angle)
#print(des1[5442])

# Feature matching with OpenCV
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x: x.distance)
output = cv.drawMatches(gray1, kp1, gray2, kp2, matches[:50], img2.copy(), flags=0)

print(matches[555].queryIdx) # trainIdx, queryIdx

# Get some matched point
i1 = matches[555].queryIdx
i2 = matches[555].trainIdx
p1 = kp1[i1].pt
p2 = kp2[i2].pt
p1 = (int(p1[0]), int(p1[1]))
p2 = (int(p2[0]), int(p2[1]))
img1 = cv.circle(img1, p1, 30, (0, 255, 0), 3)
img2 = cv.circle(img2, p2, 30, (0, 255, 0), 3)

plt.figure()
plt.gray()

plt.subplot(121)
plt.imshow(img1)

plt.subplot(122)
plt.imshow(img2)

plt.show()