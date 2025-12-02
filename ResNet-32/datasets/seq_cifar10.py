# -*-coding:utf8-*-

import torch
import torchvision
from torch.utils.data import DataLoader
import random

from datasets.single_task_dataset import SimpleDataset


class SeqCIFAR10(object):
    def __init__(self, data_path, download, class_order, tasks, batch_size, workers=0, first_task_class=None):
        self.data_path = data_path
        self.download = download
        if class_order is not None:
            self.class_order = class_order
        else:
            self.class_order = list(range(10))
        self.tasks = tasks
        self.batch_size = batch_size
        self.workers = workers
        self.first_task_class = first_task_class
        self.task_dic = self.make_task_dic()
        self.dims = torch.Size([3, 32, 32])
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
             torchvision.transforms.RandomCrop(self.dims[-2:], padding=4),
             torchvision.transforms.RandomHorizontalFlip()]
        )
        self.evaL_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))]
        )
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=True, download=self.download, transform=None)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=False, download=self.download, transform=None)

    def make_task_dic(self):
        task_dic = {}
        for i in range(self.tasks):
            task_dic[i] = []
        start = 0
        tid = 0
        if self.first_task_class is not None:
            task_dic[0] = self.class_order[0:self.first_task_class]
            start = self.first_task_class
            tid = 1
            class_per_task = int((len(self.class_order) - self.first_task_class) // (self.tasks - tid))
        else:
            class_per_task = int(len(self.class_order) // (self.tasks - tid))
        for i in range(tid, self.tasks):
            task_dic[i] = self.class_order[start:start + class_per_task]
            start += class_per_task
        return task_dic

    def get_task_loaders(self, task_id, train_size=-1):
        class_set = set(self.task_dic[task_id])
        old_class = 0
        for i in range(task_id):
            old_class += len(self.task_dic[i])
        # make class map, map real class to relative class
        class_map = {}
        for i, ci in enumerate(self.task_dic[task_id]):
            class_map[ci] = old_class + i
        train_samples = []
        for di in self.train_dataset:
            sp, lab = di
            if int(lab) in class_set:
                new_lab = class_map[int(lab)]
                train_samples.append([sp, new_lab])
        test_samples = []
        for di in self.test_dataset:
            sp, lab = di
            if int(lab) in class_set:
                new_lab = class_map[int(lab)]
                test_samples.append([sp, new_lab])
        if train_size > 0:
            train_samples = random.sample(train_samples, train_size)
        train_dataset = SimpleDataset(data=train_samples, transforms=self.transforms)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.workers, drop_last=False, shuffle=True)
        test_dataset = SimpleDataset(data=test_samples, transforms=self.evaL_transforms)
        test_loader = DataLoader(test_dataset, batch_size=100, drop_last=False)
        return train_loader, test_loader

    def get_joint_data(self, classes=-1):
        if classes == -1:
            target_classes = self.class_order
        else:
            target_classes = self.class_order[:classes]
        class_map = {}
        for i, ci in enumerate(target_classes):
            class_map[ci] = i
        class_set = set(target_classes)
        train_samples = []
        for di in self.train_dataset:
            sp, lab = di
            if int(lab) in class_set:
                new_lab = class_map[int(lab)]
                train_samples.append([sp, new_lab])
        test_samples = []
        for di in self.test_dataset:
            sp, lab = di
            if int(lab) in class_set:
                new_lab = class_map[int(lab)]
                test_samples.append([sp, new_lab])
        train_dataset = SimpleDataset(data=train_samples, transforms=self.transforms)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.workers, drop_last=False, shuffle=True)
        test_dataset = SimpleDataset(data=test_samples, transforms=self.evaL_transforms)
        test_loader = DataLoader(test_dataset, batch_size=100, drop_last=False)
        return train_loader, test_loader

    def get_raw_task_data(self, task_id):
        class_set = set(self.task_dic[task_id])
        old_class = 0
        for i in range(task_id):
            old_class += len(self.task_dic[i])
        # make class map, map real class to relative class
        class_map = {}
        for i, ci in enumerate(self.task_dic[task_id]):
            class_map[ci] = old_class + i
        train_samples = []
        for di in self.train_dataset:
            sp, lab = di
            if int(lab) in class_set:
                new_lab = class_map[int(lab)]
                train_samples.append([sp, new_lab])
        return train_samples

    def get_task_dic(self):
        return self.task_dic

    def get_transforms(self):
        return self.transforms
