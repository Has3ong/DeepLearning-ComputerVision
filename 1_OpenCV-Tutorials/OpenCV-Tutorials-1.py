import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))


image = cv2.imread(__BASE_DIR__ + '/src/louvre.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("GrayScale", gray_image)
cv2.imshow("Original", image)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('output.png', image)