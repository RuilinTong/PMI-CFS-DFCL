# -*-coding:utf8-*-

# generator model for model inversion.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class CIFARGenerator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super().__init__()

        self.z_dim = zdim
        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size):
        device = next(self.parameters()).device
        z = torch.randn(size, self.z_dim).to(device)
        X = self.forward(z)
        return X


def CIFAR_GEN():
    return CIFARGenerator(zdim=1000, in_channel=3, img_sz=32)


class CIFARGeneratorLarge(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super().__init__()

        self.z_dim = zdim
        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, 256 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(256),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size):
        device = next(self.parameters()).device
        z = torch.randn(size, self.z_dim).to(device)
        X = self.forward(z)
        return X


def CIFAR_GEN_large():
    return CIFARGeneratorLarge(zdim=1000, in_channel=3, img_sz=32)


class TinyImageNetGenerator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super().__init__()

        self.z_dim = zdim
        self.init_size = img_sz // 8
        self.l1 = nn.Sequential(nn.Linear(zdim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks3(img)
        return img

    def sample(self, size):
        device = next(self.parameters()).device
        z = torch.randn(size, self.z_dim).to(device)
        X = self.forward(z)
        return X


def TINYIMNET_GEN():
    return TinyImageNetGenerator(zdim=1000, in_channel=3, img_sz=64)


class DCMICIFARGenerator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz, prototype_sim_matrix):
        super().__init__()
        self.z_dim = zdim
        # for DCMI
        self.prototype_sim_matrix = prototype_sim_matrix
        self.num_class = prototype_sim_matrix.shape[0]
        self.emb = torch.nn.Embedding(num_embeddings=self.num_class, embedding_dim=self.z_dim)
        self.class_sim_prob = F.softmax(self.prototype_sim_matrix, dim=1)

        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim * 2, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False),
        )

    def forward(self, z, y):
        y = y / torch.norm(y, p=2, dim=1, keepdim=True)
        out = self.l1(torch.cat([z, y], dim=1))
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


def DCMICIFAR_GEN(prototype_sim):
    return DCMICIFARGenerator(
        zdim=500, in_channel=3, img_sz=32, prototype_sim_matrix=prototype_sim)


class DCMITinyImageNetGenerator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz, prototype_sim_matrix):
        super().__init__()
        self.z_dim = zdim
        # for DCMI
        self.prototype_sim_matrix = prototype_sim_matrix
        self.num_class = prototype_sim_matrix.shape[0]
        self.emb = torch.nn.Embedding(num_embeddings=self.num_class, embedding_dim=self.z_dim)
        self.class_sim_prob = F.softmax(self.prototype_sim_matrix, dim=1)

        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim * 2, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False),
        )

    def forward(self, z, y):
        y = y / torch.norm(y, p=2, dim=1, keepdim=True)
        out = self.l1(torch.cat([z, y], dim=1))
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def get_class_embedding(self, classes, device):
        """
        compute the embedding of classes
        :param classes:
        :param device:
        :return:
        """
        all_embeds = self.emb(torch.arange(self.num_class).to(device))
        class_embeds = []
        class_probs = []
        for ci in classes:
            sim_prob = torch.unsqueeze(self.class_sim_prob[ci, :], dim=1)
            class_embed = torch.sum(all_embeds * sim_prob, dim=0)
            class_embeds.append(class_embed)
            class_probs.append(torch.squeeze(sim_prob))
        return torch.stack(class_embeds, dim=0), torch.stack(class_probs, dim=0).detach()


def DCMITINYIMNET_GEN(prototype_sim):
    return DCMITinyImageNetGenerator(zdim=1000, in_channel=3, img_sz=64, prototype_sim_matrix=prototype_sim)


class Discriminator(nn.Module):
    def __init__(self, M=32):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = nn.Linear(M // 8 * M // 8 * 512, 1)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class DiscriminatorTiny(nn.Module):
    def __init__(self, M=64):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = nn.Linear(M // 8 * M // 8 * 512, 1)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
