from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
import pickle


if __name__ == "__main__":
    with open("VGG16_feature_dataset.pkl", "rb") as f:
        p = pickle.load(f)
    train_x = p["train_conv_feature"]
    test_x = p["test_conv_feature"]
    train_y = np.asarray(np.argmax(p["train_label"], axis=1), dtype=np.float32)
    test_y = np.asarray(np.argmax(p["test_label"], axis=1), dtype=np.float32)
    clf = SVC(kernel="linear")
    print(train_x.shape, train_y.shape)
    clf.fit(train_x, train_y)
    print(np.mean(test_y == clf.predict(test_x)))
    joblib.dump(clf, "svc_VGG16.m")

