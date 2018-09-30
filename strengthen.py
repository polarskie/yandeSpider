import cv2
import numpy as np
import time


class ImageStrengthen:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        np.random.seed(int(time.time()))

    def random_flip(self, img):
        if np.random.rand() < 0.5:
            img = img[::-1]
        if np.random.rand() < 0.5:
            img = img[:, ::-1]
        return img

    def random_rotate(self, img):
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        angle = int(360 * np.random.rand())
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        return img

    def generate(self, img):
        return self.random_rotate(self.random_flip(img))
