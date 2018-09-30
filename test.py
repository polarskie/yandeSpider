import keras
import threading
import time
import cv2
from BatchGenerator import BatchGenerator
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import sys
import os

batch_lock = threading.Lock()
input_X = []
input_y = []


def one_hot(y):
    w = np.max(y) + 1
    ret = np.zeros((len(y), w))
    for i, j in enumerate(y):
        ret[i][j] = 1
    return ret


def read_image_batch(path_list):
    global batch_lock
    global input_X
    input_X = []
    for path in path_list:
        img = cv2.imread(path)
        if img is None:
            print("ERROR: %s does not exist" % path)
        input_X.append(cv2.resize(cv2.imread(path), dsize=(224, 224)))
    batch_lock.release()


if __name__ == "__main__":
    # load model
    print("loading mode")
    if len(sys.argv) <= 2:
        print("must provide a test file list and a model.h5 file")
        quit()
    else:
        model = keras.models.load_model(sys.argv[2])
    # load image paths and labels
    print("loading image paths and labels")
    with open(sys.argv[1]) as f:
        lines = f.read().strip().split('\n')
    test_X_paths = [l.strip().split(' ')[0] for l in lines]
    test_y = one_hot([int(l.strip().split(' ')[1]) for l in lines])
    batch_lock.acquire()
    threading.Thread(target=read_image_batch, args=(test_X_paths, )).start()
    batch_lock.acquire()
    test_X_array = np.array(input_X)
    test_y_array = np.array(test_y)
    batch_lock.release()
    
    if os.path.exists("./false_positive"):
        os.system("mv false_positive false_positive_backup")
    os.system("mkdir false_positive")
    if os.path.exists("./false_negative"):
        os.system("mv false_negative false_negative_backup")
    os.system("mkdir false_negative")
    # train
    prediction_array = model.predict(test_X_array, batch_size=20)
    print(prediction_array)
    print(np.mean(np.argmax(prediction_array, axis=1) == np.argmax(test_y_array, axis=1)))
    mean_cross_entropy = np.mean(np.sum(-np.log2(prediction_array) * test_y_array, axis=1))
    print(mean_cross_entropy)
    prediction_1 = np.argmax(prediction_array, axis=1)
    test_y_1 = np.argmax(test_y_array, axis=1)
    F = prediction_1 != test_y_1
    P = prediction_1 == 1
    N = prediction_1 == 0
    FP_paths = [test_X_paths[i] for i, j in enumerate(F & P) if j]
    FN_paths = [test_X_paths[i] for i, j in enumerate(F & N) if j]
    for p in FP_paths:
        os.system("cp %s false_positive/" % p)
    for p in FN_paths:
        os.system("cp %s false_negative/" % p)
