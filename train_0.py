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
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from strengthen import ImageStrengthen 
import urllib3
import json


http = urllib3.PoolManager()
batch_lock = threading.Lock()
input_X = []
input_y = []
strengthenor = ImageStrengthen()
test_interval = 100
record_interval = 1000


def one_hot(y):
    w = np.max(y) + 1
    ret = np.zeros((len(y), w))
    for i, j in enumerate(y):
        ret[i][j] = 1
    return ret


def normalize_img(m):
    return np.asarray(m, dtype=np.float32) / 256.0


def read_image_batch(path_list, normalizing=True, strengthing=True):
    global batch_lock
    global input_X
    global strengthenor
    input_X = []
    for path in path_list:
        img = cv2.imread(path)
        if img is None:
            print("ERROR: %s does not exist" % path)
        if strengthing:
            img = strengthenor.generate(img)
        else:
            img = cv2.resize(img, (224, 224))
        if normalizing:
            img = normalize_img(img)
        input_X.append(img)
    batch_lock.release()


if __name__ == "__main__":
    base_kernel_size = 32
    # load model
    if len(sys.argv) < 3:
        print("must provide train and test file list")
        quit()
    if len(sys.argv) == 3:
        print("building model")
        model = Sequential()
        model.add(Conv2D(base_kernel_size, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
        # model.add(BatchNormalization(axis=3))
        model.add(Conv2D(base_kernel_size, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(base_kernel_size*2, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(base_kernel_size*2, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(base_kernel_size*4, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(base_kernel_size*8, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(base_kernel_size*16, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        # model.add(BatchNormalization(axis=1))
        model.add(Dense(2, activation='softmax'))
        for layer in model.layers:
            print(layer.output)
        # model.compile(keras.optimizers.Adam(lr=0.0005), keras.losses.categorical_crossentropy)
        model.compile(keras.optimizers.Adam(lr=0.0001), keras.losses.categorical_crossentropy)
        # model.compile(keras.optimizers.SGD(lr=0.0001, momentum=0.95, decay=1e-6), keras.losses.categorical_crossentropy)
    
        accuracies = []
        mean_cross_entropies = []
        train_loss = []
        train_loss_avg = []
    else:
        print("loading existing model")
        model = keras.models.load_model(sys.argv[3])
        if len(sys.argv) >= 5:
            with open(sys.argv[4], "rb") as f:
                p = pickle.load(f)
            accuracies = ["accs"]
            mean_cross_entropies = p["crs_ents"]
            train_loss_avg = p["losses"]
        accuracies = []
        mean_cross_entropies = []
        train_loss_avg = []
        train_loss = []

    # model.compile(keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9), keras.losses.categorical_crossentropy)
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
    threading.Thread(target=read_image_batch, args=(test_X_paths, True, False)).start()
    batch_lock.acquire()
    test_X_array = np.array(input_X)
    test_y_array = np.array(test_y)
    batch_lock.release()
    bg = BatchGenerator(len(train_y), 20)
    
    # train
    print("start training")
    batch_inds = bg.next_batch()
    batch_lock.acquire()
    best_acc = -1.0
    threading.Thread(target=read_image_batch, args=([train_X_paths[ind] for ind in batch_inds], )).start()
    for batch_i in range(10000000):
        batch_lock.acquire()
        train_y_array = np.array([train_y[ind] for ind in batch_inds])
        if False:
            for mi, m in enumerate(input_X):
                cv2.imwrite("kkjj%i.jpg" % mi, m)
            quit()
        train_X_array = np.array(input_X)
        batch_inds = bg.next_batch()
        threading.Thread(target=read_image_batch, args=([train_X_paths[ind] for ind in batch_inds], )).start()
        loss = model.train_on_batch(train_X_array, train_y_array)
        train_loss.append(float(loss))
        print("batch %i over" % batch_i)
        if batch_i % test_interval == 0:
            prediction_array = model.predict(test_X_array, batch_size=20)
            mean_cross_entropy = np.mean(np.sum(-np.log2(prediction_array) * test_y_array, axis=1))
            accuracy = np.mean(np.argmax(prediction_array, axis=1) == np.argmax(test_y_array, axis=1))
            print(mean_cross_entropy, accuracy)
            accuracies.append(accuracy)
            mean_cross_entropies.append(mean_cross_entropy)
            train_loss_avg.append(np.mean(train_loss[-test_interval:]))
            try:
                r = http.request("POST", 'http://localhost:8080/log', 
                                 body=json.dumps({"accuracies": accuracies,
                                                  "mean_cross_entropies": mean_cross_entropies,
                                                  "loss": train_loss_avg}).encode('utf-8'))
                print(r.status)
            except:
                print("ERROR: failed in sending training logs")
            if accuracy > best_acc:
                best_acc = accuracy
                model.save("batch_best.h5")
            if batch_i % record_interval == 0:
                time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
                model.save("batch_%s.h5" % time_str)
                with open("log.txt", 'wb') as f:
                    pickle.dump({"accs": accuracies, "crs_ents": mean_cross_entropies, "losses": train_loss_avg}, f)
