from sklearn.neighbors import KNeighborsClassifier as KNN
import sys
import pickle
import numpy as np


if __name__ == "__main__":
    if len(sys.argv) > 1:
        k = int(sys.argv[1])
    else:
        k = 1
    clf = KNN(n_neighbors=k, n_jobs=8)
    with open("VGG16_feature_dataset.pkl", 'rb') as f:
        p = pickle.load(f)
    train_x = p["train_conv_feature"]
    train_y = np.asarray(np.argmax(p["train_label"], axis=1), dtype=np.float32)
    test_x = p["test_conv_feature"]
    test_y = np.asarray(np.argmax(p["test_label"], axis=1), dtype=np.float32)
    clf.fit(train_x, train_y)
    predict = []
    for i in range(0, len(test_x), 100):
        predict.append(clf.predict(test_x[i:min(len(test_x), i+100)]))
        print(i)
    predict = np.concatenate(predict, axis=0)
    print(np.mean(predict == test_y))

