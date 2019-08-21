import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/apple.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = image.shape

sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

laplacian = cv2.Laplacian(image, cv2.CV_64F)
canny = cv2.Canny(image, 80, 190)


cv2.imshow('Original', image)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('Canny', canny)
cv2.waitKey(0)

cv2.destroyAllWindows()