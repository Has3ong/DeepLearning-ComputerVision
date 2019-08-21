import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/wood.jpg')

height, width = image.shape[:2]
half_height = height// 2
half_width = width// 2

T = np.float32([[1, 0, half_height], [0, 1, half_width]])
translate_image = cv2.warpAffine(image, T, (width, height))

height, width = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, .5)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5)

cropp_image = image[200: 400, 200: 400]

cv2.imshow('Original', image)
cv2.imshow('Rottation', rotated_image)
cv2.imshow('Translation', translate_image)
cv2.imshow('Scale', scaled_image)
cv2.imshow('Cropping', cropp_image)

cv2.waitKey()
cv2.destroyAllWindows()