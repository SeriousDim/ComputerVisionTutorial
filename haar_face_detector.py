import os
import cv2 as cv
import time as t

""" Сделать фото лица. По нажатию на любую кнопку сохраняет
фото лица. Положение лица определяется по каскадам Хаара
"""

cam = cv.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_id = "dima"  # подпапка папки 'images/'

tm = t.time()
faces = []
count = 0
border = 35

while True:
    _, f = cam.read()
    gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    if (t.time() - tm) >= 0.5:
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.2,
                                             minNeighbors=5,
                                             minSize=(100,100))
        tm = t.time()
    for (x, y, w, h) in faces:
        cv.rectangle(f, (x - border, y - border), (x + w + border, y + h + border), (0, 255, 0), 1)
    cv.imshow('cam', f)
    key = cv.waitKey(20)
    if key == 27:
        break
    if key != -1:
        for (x, y, w, h) in faces:
            # подпапка с именем face_id должна существовать!
            cv.imwrite("images/"+face_id+"/"+str(count)+".jpg", f[y-border: y+h+border, x-border: x+w+border])
            count += 1
            print(str(count)+".jpg saved")

cam.release()
cv.destroyAllWindows()