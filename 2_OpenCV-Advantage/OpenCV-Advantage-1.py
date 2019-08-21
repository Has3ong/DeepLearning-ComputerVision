import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/novel.jpg')

contour = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.uint8)
gray = cv2.dilate(gray, kernel, iterations = 3)

canny = cv2.Canny(gray, 10, 100)

_, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(contour, contours, -1, (0, 255, 0), 3)

cv2.imshow('original', image)
cv2.imshow('Canny', canny)
cv2.imshow('Contour', contour)

cv2.waitKey(0)
cv2.destroyAllWindows()