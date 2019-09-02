import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
from keras.models import load_model

__BASE_DIR__ = os.path.dirname(os.path.realpath(__file__))
classifier = load_model(__BASE_DIR__ + '/monkey_species_mobileNet.h5')

monkey_breeds_dict = {"[0]": "mantled_howler ",
                      "[1]": "patas_monkey",
                      "[2]": "bald_uakari",
                      "[3]": "japanese_macaque",
                      "[4]": "pygmy_marmoset ",
                      "[5]": "white_headed_capuchin",
                      "[6]": "silvery_marmoset",
                      "[7]": "common_squirrel_monkey",
                      "[8]": "black_headed_night_monkey",
                      "[9]": "nilgiri_langur"}

monkey_breeds_dict_n = {"n0": "mantled_howler ",
                        "n1": "patas_monkey",
                        "n2": "bald_uakari",
                        "n3": "japanese_macaque",
                        "n4": "pygmy_marmoset ",
                        "n5": "white_headed_capuchin",
                        "n6": "silvery_marmoset",
                        "n7": "common_squirrel_monkey",
                        "n8": "black_headed_night_monkey",
                        "n9": "nilgiri_langur"}


def draw_test(name, pred, im):
    monkey = monkey_breeds_dict[str(pred)]
    BLACK = [0, 0, 0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100, cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, monkey, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0, len(folders))
    path_class = folders[random_directory]
    print("Result - " + monkey_breeds_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0, len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path + "/" + image_name)


for i in range(0, 10):
    input_im = getRandomImage(__BASE_DIR__ + "/src/data/10-monkey-species/validation/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    input_im = cv2.resize(input_im, (224, 224), interpolation=cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1, 224, 224, 3)

    res = np.argmax(classifier.predict(input_im, 1, verbose=0), axis=1)

    draw_test("Prediction", res, input_original)
    cv2.waitKey(0)

cv2.destroyAllWindows()