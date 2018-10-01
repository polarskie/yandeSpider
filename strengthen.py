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

    def random_rotate(self, img, angle):
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        return img

    def generate(self, img):
        angle = 360 * np.random.rand()
        contain_size = int(np.ceil(224. * (np.sin(angle * np.pi / 180.) + np.cos(angle * np.pi / 180.))))
        img = cv2.resize(img, dsize=(contain_size, contain_size))
        img = self.random_rotate(self.random_flip(img), angle)
        img = cv2.resize(img, dsize=(224, 224))
        return img
