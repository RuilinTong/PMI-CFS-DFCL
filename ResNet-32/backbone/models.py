# -*-coding:utf8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu


class ConvNet(nn.Module):
    def __init__(self, output_dim):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.dp1 = torch.nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dp2 = torch.nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 128)
        self.dp3 = torch.nn.Dropout(0.5)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc2(x)
        return x

    def embed(self, x):
        x = F.relu(self.dp1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.dp2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.dp3(self.fc1(x)))
        return x
