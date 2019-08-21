import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/board.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

for c in corners:
    x, y = c[0]
    x, y = int(x), int(y)
    cv2.rectangle(image, (x-10, y-10), (x+10, y+10), (0, 0, 255), 2)


cv2.imshow('Corner', image)
cv2.waitKey(0)
cv2.destroyAllWindows()