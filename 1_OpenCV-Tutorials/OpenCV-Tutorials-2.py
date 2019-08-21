import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))

image = cv2.imread(__BASE_DIR__ + '/src/louvre.jpg')

# 0, 0 pixel BGR Values
B, G, R = image[0, 0]
#94, 114, 162
print(B, G, R)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
value = gray_image[0, 0]
# 126
print(value)

# 426, 640, 3
print(image.shape)

# 426, 640
print(gray_image.shape)


# HSV Channel
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV image', hsv_image)
cv2.imshow('Hue channel', hsv_image[:, :, 0])
cv2.imshow('Saturation channel', hsv_image[:, :, 1])
cv2.imshow('Value channel', hsv_image[:, :, 2])


# RGB Channel
B, G, R = cv2.split(image)
zeros = np.zeros(image.shape[:2], dtype = "uint8")

cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()