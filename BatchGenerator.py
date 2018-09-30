import numpy as np
import time


class BatchGenerator:
    def __init__(self, total_size, batch_size):
        np.random.seed(int(time.time()))
        assert batch_size <= total_size
        self.total_size = total_size
        self.batch_size = batch_size
        self.tmp_ind = 0
        self.inds = np.arange(total_size)
        self.epoch = 0
        np.random.shuffle(self.inds)

    def next_batch(self):
        if self.tmp_ind + self.batch_size > self.total_size:
            self.tmp_ind = 0
            np.random.shuffle(self.inds)
            self.epoch += 1
        self.tmp_ind += self.batch_size
        return self.inds[self.tmp_ind-self.batch_size:self.tmp_ind]
    