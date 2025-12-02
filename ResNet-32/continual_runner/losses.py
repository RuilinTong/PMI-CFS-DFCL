# -*-coding:utf8-*-

# for losses in continual learning

import torch
import torch.nn.functional as F
import numpy as np


class LocalCELoss(torch.nn.Module):
    def __init__(self, start_class, end_class):
        super(LocalCELoss, self).__init__()
        self.start_class = start_class
        self.end_class = end_class
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = x[:, self.start_class:]
        y = y - self.start_class
        loss = self.ce_loss(x, y)
        return loss


class WeightedFeatureLoss(torch.nn.Module):
    """
    used for ABD feature distillation
    """
    def __init__(self, head):
        super(WeightedFeatureLoss, self).__init__()
        self.head = head
        self.head.eva()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x, y):
        x = self.head(x)
        y = self.head(y)
        loss = self.mse_loss(x, y)
        return loss


class RelationKDLoss(torch.nn.Module):
    """
    used for relation knowledge distillation
    """
    def __init__(self, input_dim, output_dim):
        super(RelationKDLoss, self).__init__()
        self.linear_old = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
        self.linear_new = torch.nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x, y):
        student, teacher = self.linear_new(x), self.linear_old(y)

        td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.l1_loss(s_angle, t_angle)
        return loss


class FeatureRelationKDLoss(torch.nn.Module):
    def __init__(self):
        super(FeatureRelationKDLoss, self).__init__()
        self.flatten = torch.nn.Flatten()

    def forward(self, x, y):
        flat_x = self.flatten(x)
        flat_y = self.flatten(y)

        rolled_y = torch.roll(flat_y, shifts=1, dims=0)
        norm_y = F.normalize(flat_y, p=2, dim=1)
        norm_roll_y = F.normalize(rolled_y, p=2, dim=1)
        t_angle = torch.sum(norm_y * norm_roll_y, dim=1)

        rolled_x = torch.roll(flat_x, shifts=1, dims=0)
        norm_x = F.normalize(flat_x, p=2, dim=1)
        norm_roll_x = F.normalize(rolled_x, p=2, dim=1)
        s_angle = torch.sum(norm_x * norm_roll_x, dim=1)

        loss = F.l1_loss(s_angle, t_angle)
        return loss


def feature_distance(x, y, std_x):
    """
    compute feature distance used in ABD
    :param x: numpy array [1, feature_dim], mean feature of model1
    :param y: numpy array [1, feature_dim], mean feature of model1
    :param std_x: std feature of model 1
    :return:
    """
    diff = (x - y) / (std_x + 1e-4)
    dist = np.linalg.norm(diff, 2)
    return dist


def cosine_distance(x, y):
    """
    compute cosine distance between mean of two vector sets
    this may be used for computing cosine distance on logit
    :param x: numpy array [batch_size, feature_dim], feature of model1
    :param y: numpy array [batch_size, feature_dim], feature of model1
    :return:
    """
    norm_x = np.linalg.norm(x, ord=2) + 1e-8
    norm_y = np.linalg.norm(y, ord=2) + 1e-8
    dist = np.sum((x / norm_x) * (y / norm_y))
    return 1 - dist


def l2_distance(x, y):
    """
    compute L2 distance between mean of two vector sets
    :param x: numpy array [batch_size, feature_dim], feature of model1
    :param y: numpy array [batch_size, feature_dim], feature of model1
    :return:
    """
    dist = np.linalg.norm((x - y), ord=2)
    return dist
