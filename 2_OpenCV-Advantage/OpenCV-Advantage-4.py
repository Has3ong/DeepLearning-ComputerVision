import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__+'/src/people.jpg')
haar = image.copy()

face_cascade = cv2.CascadeClassifier(__BASE_DIR__ + '/src/haarcascades/haarcascade_frontalface_default.xml')

face_rects = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)

for (x, y, w, h) in face_rects:
    cv2.rectangle(haar, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Original', image)
cv2.imshow('HAAR Cascade', haar)
cv2.waitKey()
cv2.destroyAllWindows()
