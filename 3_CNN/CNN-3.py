from keras.models import load_model
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import random
import os

label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))

model = load_model(__BASE_DIR__ + '/my-cifar-model.h5')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

for i in range(x_test.shape[0]):
    if model.predict_classes(x_test[i].reshape((1, 32, 32, 3)))[0] != y_test[i][0]:
        digit = x_test[i].reshape((32, 32, 3))
        digit = digit.astype('float32') * 255
        digit = digit.astype('uint8')

        plt.imshow(digit)
        plt.xlabel("predict image : " + label[int(model.predict_classes(x_test[i].reshape((1, 32, 32, 3))))] + "\t" + "validate image : "  + label[y_test[i][0]] )
        plt.show()
