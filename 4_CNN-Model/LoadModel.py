import keras
import cv2
import os
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from keras.applications import vgg16, inception_v3, resnet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))

# Loads the VGG16 model
vgg_model = vgg16.VGG16(weights='imagenet')
# Loads the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')
# Loads the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet')

mypath = __BASE_DIR__ + '/src/images/'
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

def draw_test(name, preditions, input_im):
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[1]+300 ,cv2.BORDER_CONSTANT,value=BLACK)
    img_width = input_im.shape[1]
    for (i,predition) in enumerate(preditions):
        string = str(predition[1]) + " " + str(predition[2])
        cv2.putText(expanded_image,str(name),(img_width + 50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),1)
        cv2.putText(expanded_image,string,(img_width + 50,50+((i+1)*50)),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
        print(string)
    cv2.imshow(name, expanded_image)


for file in file_names:

    img = image.load_img(mypath + file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    img = image.load_img(mypath + file, target_size=(299, 299))
    v3 = image.img_to_array(img)
    v3 = np.expand_dims(v3, axis=0)
    v3 = preprocess_input(v3)

    img2 = cv2.imread(mypath + file)
    imageL = cv2.resize(img2, None, fx=.5, fy=.5, interpolation=cv2.INTER_CUBIC)

    # Get VGG16 Predictions
    print("VGG16 Prediction")
    preds_vgg_model = vgg_model.predict(x)
    preditions_vgg = decode_predictions(preds_vgg_model, top=3)[0]
    draw_test("VGG16 Predictions", preditions_vgg, imageL)


    # Get ResNet50 Predictions
    print("\nResNet50 Predictions")
    preds_resnet = resnet_model.predict(x)
    preditions_resnet = decode_predictions(preds_resnet, top=3)[0]
    draw_test("ResNet50 Predictions", preditions_resnet, imageL)


    # Get Inception_V3 Predictions
    print("\nInception_V3 Predictions")
    preds_inception = inception_model.predict(v3)
    preditions_inception = decode_predictions(preds_inception, top=3)[0]
    draw_test("Inception_V3 Predictions", preditions_inception, imageL)

    cv2.waitKey(0)

cv2.destroyAllWindows()