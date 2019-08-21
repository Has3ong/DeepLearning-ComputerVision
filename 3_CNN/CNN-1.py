from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import random

batch_size = 128
epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# (60000,28,28) to (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


for i in range(1000):
    r = random.randint(0, len(x_test) -1)
    if int(model.predict_classes(x_test[r].reshape((1, 28, 28, 1)))) != list(y_test[r]).index(1.0):
        digit = x_test[r].reshape((28, 28))
        digit = digit.astype('float32') * 255
        digit = digit.astype('uint8')

        plt.imshow(digit, cmap=plt.cm.binary)
        plt.show()
        print("예상 정답값 : ", int(model.predict_classes(x_test[r].reshape((1, 28, 28, 1)))))
        print("실제 정답값 : ", list(y_test[r]).index(1.0))
