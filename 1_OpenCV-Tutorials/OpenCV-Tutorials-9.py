import cv2
import numpy as np
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
image = cv2.imread(__BASE_DIR__ + '/src/novel.jpg')


# (460, 46) (843, 42) (473, 644) (885, 638)
before = np.float32([[460, 46], [843, 42], [473, 644], [885, 638]])

after = np.float32([[0, 0], [300, 0], [0, 420], [300, 420]])

matrix = cv2.getPerspectiveTransform(before, after)

warped = cv2.warpPerspective(image, matrix, (300, 420))

cv2.imshow('Warped', warped)
cv2.imshow('Orginal', image)
cv2.waitKey(0)

cv2.destroyAllWindows()