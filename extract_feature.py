import keras
import threading
import time
import cv2
from BatchGenerator import BatchGenerator
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import sys
import pickle

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
    input_tensor = keras.Input(shape=(224, 224, 3))
    conv_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), input_tensor=input_tensor)
    for layer in conv_model.layers:
        layer.trainable = False
    conv_feature_output = conv_model.output
    x = keras.layers.Flatten()(conv_feature_output)
    # x = keras.layers.Dense(units=256, activation="relu")(x)
    # x = keras.layers.Dense(units=128, activation="relu")(x)
    # x = keras.layers.Dense(units=2, activation="softmax")(x)
    model = keras.Model(inputs=input_tensor, outputs=x)
    model.compile(keras.optimizers.Adam(lr=0.0001), keras.losses.categorical_crossentropy)
    # model.compile(keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9), keras.losses.categorical_crossentropy)

    # load image paths and labels
    print("loading image paths and labels")
    with open("train_list.txt") as f:
        lines = f.read().strip().split('\n')
    train_X_paths = [l.strip().split(' ')[0] for l in lines]
    train_y = one_hot([int(l.strip().split(' ')[1]) for l in lines])
    with open("test_list.txt") as f:
        lines = f.read().strip().split('\n')
    test_X_paths = [l.strip().split(' ')[0] for l in lines]
    test_y = one_hot([int(l.strip().split(' ')[1]) for l in lines])
    batch_lock.acquire()
    threading.Thread(target=read_image_batch, args=(test_X_paths, )).start()
    batch_lock.acquire()
    test_X_array = np.array(input_X)
    test_y_array = np.array(test_y)
    batch_lock.release()
    
    # extract features
    print("start training")
    batch_lock.acquire()
    accuracies = []
    mean_cross_entropies = []
    tmp_ind = 0
    threading.Thread(target=read_image_batch, args=(train_X_paths[tmp_ind:min(len(train_X_paths), tmp_ind+20)], )).start()
    tmp_ind += 20
    train_conv_feature_list = []
    while tmp_ind < len(train_X_paths):
        batch_lock.acquire()
        train_X_array = np.array(input_X)
        threading.Thread(target=read_image_batch, args=(train_X_paths[tmp_ind:min(len(train_X_paths), tmp_ind+20)], )).start()
        tmp_ind += 20
        train_conv_feature_list.append(model.predict_on_batch(train_X_array))
        print(train_conv_feature_list[-1].shape)
        print("index %i over" % tmp_ind)
    train_X_array = np.array(input_X)
    train_conv_feature_list.append(model.predict_on_batch(train_X_array))

    test_conv_feature = model.predict(test_X_array, batch_size=20)
    with open("VGG16_feature_dataset.pkl", 'wb') as f:
        pickle.dump({"train_conv_feature": np.concatenate(train_conv_feature_list, axis=0),
                     "train_label": np.array(train_y),
                     "test_conv_feature": test_conv_feature,
                     "test_label": np.array(test_y)}, f)
