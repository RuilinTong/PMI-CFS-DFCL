# -*-coding:utf8-*-

import random
import os
import torch
from torch.utils.data import DataLoader
import torchvision
from PIL import Image

from datasets.single_task_dataset import SimpleDataset


class FullTinyImageNetDataset(object):
    def __init__(self, data_path, class_order):
        self.data_path = data_path
        self.class_order = class_order
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

    def get_train_loader(self, batch_size, workers=-1, transforms=None, samples=-1):
        wnid_file = os.path.join(self.data_path, 'wnids.txt')
        with open(wnid_file, 'r', encoding='utf8') as fr:
            lines = fr.read().strip().split('\n')
        wnids = {k: v for v, k in enumerate(lines)}  # class to label
        train_path = os.path.join(self.data_path, 'train')
        class_map = {}
        for i, ci in enumerate(self.class_order):
            class_map[ci] = i
        train_samples = []
        for k in wnids.keys():
            lab = wnids[k]
            class_path = os.path.join(train_path, k, 'images')
            for fi in os.listdir(class_path):
                img_file = os.path.join(class_path, fi)
                img = Image.open(img_file).convert("RGB")
                new_lab = class_map[lab]
                train_samples.append([img, new_lab])
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
        wnid_file = os.path.join(self.data_path, 'wnids.txt')
        with open(wnid_file, 'r', encoding='utf8') as fr:
            lines = fr.read().strip().split('\n')
        wnids = {k: v for v, k in enumerate(lines)}  # class to label
        val_path = os.path.join(self.data_path, 'val')
        val_file = os.path.join(val_path, 'val_annotations.txt')
        with open(val_file, 'r', encoding='utf8') as fr:
            lines = fr.read().strip().split('\n')
        class_map = {}
        for i, ci in enumerate(self.class_order):
            class_map[ci] = i
        test_samples = []
        for line in lines:
            fname, wnid = line.split("\t")[:2]
            lab = wnids[wnid]
            new_lab = class_map[lab]
            img_file = os.path.join(val_path, 'images', fname)
            img = Image.open(img_file).convert("RGB")
            test_samples.append([img, new_lab])
        test_dataset = SimpleDataset(data=test_samples, transforms=self.evaL_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)
        return test_loader

    def get_transforms(self, train=True):
        if train:
            return self.transforms
        else:
            return self.evaL_transforms
