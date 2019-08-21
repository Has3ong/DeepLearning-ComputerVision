import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/comming.jpg')

kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(image, kernel, iterations = 1)
dilation = cv2.dilate(image, kernel, iterations = 1)

cv2.imshow('Original', image)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()