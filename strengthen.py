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
    
    def random_crop(self, img):
        h, w = img.shape[:2]
        horizontal_discard = int(0.2 * np.random.rand() * w)
        left_discard = int(np.random.rand() * horizontal_discard)
        right_discard = horizontal_discard - left_discard
        vertical_discard = int(0.2 * np.random.rand() * h)
        top_discard = int(np.random.rand() * vertical_discard)
        bottom_discard = vertical_discard - top_discard
        # print(h, w, top_discard, bottom_discard, left_discard, right_discard)
        img = img[top_discard:h if bottom_discard == 0 else -bottom_discard,
                  left_discard:w if right_discard == 0 else -right_discard, :]
        return img

    def random_crop_digitized(self, img):
        h, w = img.shape[:2]
        margin_h = int(h * 0.1)
        margin_w = int(w * 0.1)
        img = img[(np.random.rand() > 0.5) * margin_h:h-(np.random.rand() > 0.5) * margin_h,
                  (np.random.rand() > 0.5) * margin_w:w-(np.random.rand() > 0.5) * margin_w, :]
        return img
    
    def generate(self, img):
        # img = self.random_crop(img)
        img = self.random_crop_digitized(img)
        # angle = 360 * np.random.rand()
        angle = int(np.random.rand() / 0.125) * 45
        ang = angle
        while ang > 90.:
            ang = ang - 90.
        contain_size = int(np.ceil(224. * (np.sin(ang * np.pi / 180.) + np.cos(ang * np.pi / 180.))))
        # print(ang, contain_size)
        img = cv2.resize(img, dsize=(contain_size, contain_size))
        img = self.random_rotate(self.random_flip(img), angle)
        img = img[(contain_size-224)//2:(contain_size-224)//2+224,
                  (contain_size-224)//2:(contain_size-224)//2+224, :]
        return img
