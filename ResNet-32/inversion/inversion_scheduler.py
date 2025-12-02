# -*-coding:utf8-*-

import numpy as np


class CosineLRScheduler(object):
    def __init__(self, optimizer, warmup, total_epoch, base_lr):
        self.optimizer = optimizer
        self.warmup = warmup
        self.total_epoch = total_epoch
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            e = epoch - self.warmup
            es = self.total_epoch - self.warmup
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
