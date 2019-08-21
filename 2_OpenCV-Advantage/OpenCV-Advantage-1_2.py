import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/frame.png')

contour = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(contour, (x, y), (x + w, y + h), (0, 0, 255), 2)

for c in contours:
    accuracy = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    cv2.drawContours(contour, [approx], 0, (0, 255, 0), 2)

cv2.imshow('Original', image)
cv2.imshow('Contour', contour)

cv2.waitKey(0)
cv2.destroyAllWindows()