from keras.applications import VGG16
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))

img_rows = 224
img_cols = 224

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))


for layer in vgg16.layers:
    layer.trainable = False

num_classes = 17

top_model = vgg16.output
top_model = Flatten(name="flatten")(top_model)
top_model = Dense(256, activation="relu")(top_model)
top_model = Dropout(0.3)(top_model)
top_model = Dense(num_classes, activation="softmax")(top_model)
model = Model(inputs=vgg16.input, outputs=top_model)

model.summary()

train_data_dir = __BASE_DIR__ + '/src/data/flowers/train'
validation_data_dir = __BASE_DIR__ + '/src/data/flowers/validation'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_batchsize = 16
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)


checkpoint = ModelCheckpoint(__BASE_DIR__ + "/myflowers_vgg.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# Note we use a very small learning rate
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

nb_train_samples = 1190
nb_validation_samples = 170
epochs = 25
batch_size = 32

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

model.save(__BASE_DIR__ + "/myflowers_vgg.h5")