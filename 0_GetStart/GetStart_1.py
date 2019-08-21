import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))

classifier = load_model(__BASE_DIR__ + '/model/mnist_simple_cnn.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def draw_test(name, pred, input_im):
    BLACK = [0, 0, 0]
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (120, 90), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2)
    cv2.imshow(name, expanded_image)

for i in range(0, 10):
    rand = np.random.randint(0, len(x_test))
    input_im = x_test[rand]

    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    input_im = input_im.reshape(1, 28, 28, 1)

    res = str(classifier.predict_classes(input_im, 1, verbose=0)[0])
    draw_test("Prediction", res, imageL)
    cv2.waitKey(0)

cv2.destroyAllWindows()