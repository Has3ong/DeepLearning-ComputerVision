import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/chess.jpg')

image_line = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edges = cv2.Canny(gray, 150, 230)

lines = cv2.HoughLines(edges, 1, np.pi/180, 160)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image_line, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Original", image)
cv2.imshow("edges", edges)
cv2.imshow('Hough Lines', image_line)
cv2.waitKey(0)
cv2.destroyAllWindows()