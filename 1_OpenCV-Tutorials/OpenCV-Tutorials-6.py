import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/changdeokgung.jpg')

ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow('Original', image)
cv2.imshow('Threshold Binary', thresh1)
cv2.imshow('Threshold Binary Inverse', thresh2)
cv2.imshow('THRESH TRUNC', thresh3)
cv2.imshow('THRESH TOZERO', thresh4)
cv2.imshow('THRESH TOZERO INV', thresh5)
cv2.waitKey(0)

cv2.destroyAllWindows()