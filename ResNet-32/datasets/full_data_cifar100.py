# -*-coding:utf8-*-

import random
import torch
from torch.utils.data import DataLoader
import torchvision

from datasets.single_task_dataset import SimpleDataset


class FullCIFAR100Dataset(object):
    def __init__(self, data_path, class_order):
        self.data_path = data_path
        self.class_order = class_order
        self.dims = torch.Size([3, 32, 32])
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
             torchvision.transforms.RandomCrop(self.dims[-2:], padding=4),
             torchvision.transforms.RandomHorizontalFlip()]
        )
        self.evaL_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        )

    def get_train_loader(self, batch_size, workers=-1, transforms=None, samples=-1):
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_path, train=True, download=True, transform=None)
        train_samples = []
        class_map = {}
        for i, ci in enumerate(self.class_order):
            class_map[ci] = i
        for di in train_dataset:
            sp, lab = di
            new_lab = class_map[int(lab)]
            train_samples.append([sp, new_lab])
        if samples > 0:
            train_samples = random.sample(train_samples, samples)
        train_dataset = SimpleDataset(
            data=train_samples,
            transforms=self.transforms if transforms is None else transforms
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=workers, drop_last=False, shuffle=True)
        return train_loader

    def get_test_loader(self, batch_size):
        test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_path, train=False, download=True, transform=None)
        test_samples = []
        class_map = {}
        for i, ci in enumerate(self.class_order):
            class_map[ci] = i
        for di in test_dataset:
            sp, lab = di
            new_lab = class_map[int(lab)]
            test_samples.append([sp, new_lab])
        train_dataset = SimpleDataset(data=test_samples, transforms=self.evaL_transforms)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
        return train_loader

    def get_transforms(self, train=True):
        if train:
            return self.transforms
        else:
            return self.evaL_transforms
