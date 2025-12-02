# -*-coding:utf8-*-

import os
import random
import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image

from datasets.single_task_dataset import SimpleDataset


class SeqTinyImageNet(object):
    def __init__(self, data_path, class_order, tasks, batch_size, workers=0, first_task_class=None):
        self.data_path = data_path
        self.class_order = class_order
        self.tasks = tasks
        self.batch_size = batch_size
        self.workers = workers
        self.first_task_class = first_task_class
        # load training data
        wnid_file = os.path.join(self.data_path, 'wnids.txt')
        with open(wnid_file, 'r', encoding='utf8') as fr:
            lines = fr.read().strip().split('\n')
        self.wnids = {k: v for v, k in enumerate(lines)}  # class to label
        train_path = os.path.join(self.data_path, 'train')
        self.train_samples = {}
        for k in self.wnids.keys():
            lab = self.wnids[k]
            self.train_samples[lab] = []
            class_path = os.path.join(train_path, k, 'images')
            for fi in os.listdir(class_path):
                img_file = os.path.join(class_path, fi)
                img = Image.open(img_file).convert("RGB")
                self.train_samples[lab].append([img, lab])
        # load test data
        val_path = os.path.join(self.data_path, 'val')
        val_file = os.path.join(val_path, 'val_annotations.txt')
        with open(val_file, 'r', encoding='utf8') as fr:
            lines = fr.read().strip().split('\n')
        self.test_samples = {}
        for k in self.wnids.keys():
            lab = self.wnids[k]
            self.test_samples[lab] = []
        for line in lines:
            fname, wnid = line.split("\t")[:2]
            lab = self.wnids[wnid]
            img_file = os.path.join(val_path, 'images', fname)
            img = Image.open(img_file).convert("RGB")
            self.test_samples[lab].append([img, lab])
        # transforms
        self.dims = torch.Size([3, 64, 64])
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(64),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                (0.4803, 0.4481, 0.3976), (0.2764, 0.2688, 0.2816)),
             torchvision.transforms.RandomCrop(self.dims[-2:], padding=4),
             torchvision.transforms.RandomHorizontalFlip()]
        )
        self.evaL_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(64),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                (0.4803, 0.4481, 0.3976), (0.2764, 0.2688, 0.2816))]
        )
        # make task dic
        self.task_dic = self.make_task_dic()

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
        old_class = 0
        for i in range(task_id):
            old_class += len(self.task_dic[i])
        # make class map, map real class to relative class
        class_map = {}
        for i, ci in enumerate(self.task_dic[task_id]):
            class_map[ci] = old_class + i
        train_samples = []
        test_samples = []
        for ci in self.task_dic[task_id]:
            for di in self.train_samples[ci]:
                sp, lab = di
                new_di = [sp, class_map[ci]]
                train_samples.append(new_di)
            for di in self.test_samples[ci]:
                sp, lab = di
                new_di = [sp, class_map[ci]]
                test_samples.append(new_di)
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
        train_samples = []
        test_samples = []
        for ci in target_classes:
            for di in self.train_samples[ci]:
                sp, lab = di
                new_di = [sp, class_map[ci]]
                train_samples.append(new_di)
            for di in self.test_samples[ci]:
                sp, lab = di
                new_di = [sp, class_map[ci]]
                test_samples.append(new_di)
        train_dataset = SimpleDataset(data=train_samples, transforms=self.transforms)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.workers, drop_last=False, shuffle=True)
        test_dataset = SimpleDataset(data=test_samples, transforms=self.evaL_transforms)
        test_loader = DataLoader(test_dataset, batch_size=100, drop_last=False)
        return train_loader, test_loader

    def get_raw_task_data(self, task_id):
        old_class = 0
        for i in range(task_id):
            old_class += len(self.task_dic[i])
        # make class map, map real class to relative class
        class_map = {}
        for i, ci in enumerate(self.task_dic[task_id]):
            class_map[ci] = old_class + i
        train_samples = []
        for ci in self.task_dic[task_id]:
            for di in self.train_samples[ci]:
                sp, lab = di
                new_di = [sp, class_map[ci]]
                train_samples.append(new_di)
        return train_samples

    def get_task_dic(self):
        return self.task_dic

    def get_transforms(self):
        return self.transforms
