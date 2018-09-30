import keras
import threading
import time
import cv2
from BatchGenerator import BatchGenerator
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
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
    if len(sys.argv) < 3:
        print("must provide train and test file list")
        quit()
    if len(sys.argv) == 3:
        input_tensor = keras.Input(shape=(224, 224, 3))
        conv_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3), input_tensor=input_tensor)
        for layer in conv_model.layers:
            layer.trainable = False
        conv_feature_output = conv_model.output
        x = keras.layers.Flatten()(conv_feature_output)
        x = keras.layers.Dense(units=256, activation="relu")(x)
        # x = keras.layers.Dense(units=128, activation="relu")(x)
        x = keras.layers.Dense(units=2, activation="softmax")(x)
        model = keras.Model(inputs=input_tensor, outputs=x)
        model.compile(keras.optimizers.Adam(lr=0.0001), keras.losses.categorical_crossentropy)
        # model.compile(keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9), keras.losses.categorical_crossentropy)
    else:
        model = keras.models.load_model(sys.argv[3])
    # load image paths and labels
    print("loading image paths and labels")
    with open(sys.argv[1]) as f:
        lines = f.read().strip().split('\n')
    train_X_paths = [l.strip().split(' ')[0] for l in lines]
    train_y = one_hot([int(l.strip().split(' ')[1]) for l in lines])
    with open(sys.argv[2]) as f:
        lines = f.read().strip().split('\n')
    test_X_paths = [l.strip().split(' ')[0] for l in lines]
    test_y = one_hot([int(l.strip().split(' ')[1]) for l in lines])
    print(len(test_y))
    batch_lock.acquire()
    threading.Thread(target=read_image_batch, args=(test_X_paths, )).start()
    batch_lock.acquire()
    test_X_array = np.array(input_X)
    test_y_array = np.array(test_y)
    batch_lock.release()
    bg = BatchGenerator(len(train_y), 20)
    
    # train
    print("start training")
    batch_inds = bg.next_batch()
    batch_lock.acquire()
    accuracies = []
    mean_cross_entropies = []
    threading.Thread(target=read_image_batch, args=([train_X_paths[ind] for ind in batch_inds], )).start()
    for batch_i in range(10000):
        batch_lock.acquire()
        train_y_array = np.array([train_y[ind] for ind in batch_inds])
        train_X_array = np.array(input_X)
        batch_inds = bg.next_batch()
        threading.Thread(target=read_image_batch, args=([train_X_paths[ind] for ind in batch_inds], )).start()
        model.train_on_batch(train_X_array, train_y_array)
        print("batch %i over" % batch_i)
        if batch_i % 100 == 0:
            prediction_array = model.predict(test_X_array, batch_size=20)
            mean_cross_entropy = np.mean(np.sum(-np.log2(prediction_array) * test_y_array, axis=1))
            accuracy = np.mean(np.argmax(prediction_array, axis=1) == np.argmax(test_y_array, axis=1))
            print(mean_cross_entropy, accuracy)
            accuracies.append(accuracy)
            mean_cross_entropies.append(mean_cross_entropy)
            if batch_i % 100 == 0:
                time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
                model.save("batch_%s.h5" % time_str)
                with open("log_%s.txt" % time_str, 'wb') as f:
                    pickle.dump({"accs": accuracies, "crs_ents": mean_cross_entropies}, f)
