import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/wall.jpg')

value = np.ones(image.shape, dtype = "uint8") * 120
bright = cv2.add(image, value)
dark = cv2.subtract(image, value)


kernel = np.ones((3, 3), np.float32) / 15
blurr = cv2.filter2D(image, -1, kernel)

Gaussian = cv2.GaussianBlur(image, (7, 7), 0)


kernel = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])
sharpened = cv2.filter2D(image, -1, kernel)

cv2.imshow("Original", image)
cv2.imshow("bright", bright)
cv2.imshow("dark", dark)
cv2.imshow("Blurring", blurr)
cv2.imshow("Gaussain Blurring", Gaussian)
cv2.imshow("sharpened", sharpened)


cv2.waitKey()
cv2.destroyAllWindows()