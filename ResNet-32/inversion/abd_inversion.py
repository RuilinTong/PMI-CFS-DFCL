# -*-coding:utf8-*-
# model inversion functions

import copy
import os
import torch
import torch.nn.functional as F
import math
import pickle
import random

from inversion import generators
from inversion import functions


class ModelInversionRunner(object):
    def __init__(self, generator_type, model, lr, train_steps, batch_size, tau, alpha_pr, alpha_rf,
                 local_path=None, flip_rate=0.0):
        self.local_path = local_path
        if self.local_path is not None and (not os.path.exists(self.local_path)):
            os.makedirs(self.local_path)
        # build generator
        if generator_type == 'seq_cifar100':
            self.generator = generators.CIFAR_GEN()
        elif generator_type == 'seq_tinyimagenet':
            self.generator = generators.TINYIMNET_GEN()
        else:
            raise ValueError('Invalid generator type')
        self.model = model
        self.lr = lr
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.tau = tau  # temperature
        self.alpha_pr = alpha_pr
        self.alpha_rf = alpha_rf
        self.on_cuda = False
        self.flip_rate = flip_rate
        # build hooks
        self.stat_hooks = []
        # build Gaussian kernel
        self.gaussian_kernel = functions.Gaussiansmoothing(channels=3, kernel_size=5, sigma=1)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.slt_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def update_model(self, model):
        if self.on_cuda:
            self.model.cpu()
        self.model = model
        if self.on_cuda:
            self.model.cuda()

    def register_hooks(self):
        for n, mi in self.model.named_modules():
            if isinstance(mi, torch.nn.BatchNorm2d):
                self.stat_hooks.append(functions.BNInputHook(module=mi))

    def remove_hooks(self):
        for hi in self.stat_hooks:
            hi.remove_hook()
        self.stat_hooks.clear()

    def criterion_pr(self, inputs):
        input_pad = F.pad(inputs, (2, 2, 2, 2), mode="reflect")
        input_smooth = self.gaussian_kernel(input_pad)
        return F.mse_loss(inputs, input_smooth)

    def train_generator(self):
        functions.unfreeze(self.generator)
        # build optimizer
        opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        # build hooks:
        self.register_hooks()
        losses = {}
        # iteration
        for i in range(self.train_steps):
            inputs = self.generator.sample(self.batch_size)
            if self.flip_rate > 0:
                inputs = self.random_flip_inputs(inputs=inputs)
            output = self.model(inputs)
            target = output.data.argmax(dim=1)
            # compute losses
            l_ce = self.ce_loss(output / self.tau, target)
            l_cb = class_balance_loss(output=output)
            l_stat = torch.stack([h.stats_regularization() for h in self.stat_hooks]).mean()
            l_blur = self.criterion_pr(inputs=inputs)
            loss = l_ce + l_cb + self.alpha_pr * l_blur + self.alpha_rf * l_stat
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 1000 == 0:
                print('finish training step:', i)
                print('ce loss:', l_ce.item(), 'cb loss:', l_cb.item(), 'stat loss:',
                      l_stat.item(), 'smooth loss:', l_blur.item(), 'total loss:', loss.item())
            if i % 10 == 0:
                losses[i] = {'ce_loss': l_ce.item(), 'cb_loss': l_cb.item(), 'stat_loss': l_stat.item(),
                             'smooth_loss': l_blur.item(), 'total_loss:': loss.item()}
        functions.freeze(self.generator)
        self.remove_hooks()
        if self.local_path is not None:  # dump loss during training
            classes = self.model.head.num_classes
            loss_file = os.path.join(self.local_path, 'inv_train_logs_' + str(classes) + '.pkl')
            with open(loss_file, 'wb') as fw:
                pickle.dump(losses, fw)

    def get_data(self, batch_size, return_loss=False, return_logit=True):
        with torch.no_grad():
            _ = self.model.eval() if self.model.training else None
            batch_size = self.batch_size if batch_size is None else batch_size
            inputs = self.generator.sample(batch_size)
            out_logit = self.model(inputs)
            target = out_logit.argmax(dim=1)
            if return_loss:
                if return_logit:
                    return inputs, target, out_logit, self.slt_loss(out_logit, target)
                else:
                    return inputs, target, self.slt_loss(out_logit, target)
            else:
                if return_logit:
                    return inputs, target, out_logit
                else:
                    return inputs, target

    def cuda(self):
        self.model.cuda()
        self.generator.cuda()
        self.on_cuda = True

    def cpu(self):
        self.model.cpu()
        self.generator.cpu()
        self.on_cuda = False

    def save_models(self):
        if self.local_path is None:
            return
        classes = self.model.head.num_classes
        save_model = copy.deepcopy(self.model).cpu().eval()
        save_file = os.path.join(self.local_path, 'inv_model_' + str(classes) + '.pkl')
        torch.save(save_model, save_file)
        del save_model
        save_generator = copy.deepcopy(self.generator).cpu().eval()
        save_file = os.path.join(self.local_path, 'inv_generator_' + str(classes) + '.pkl')
        torch.save(save_generator, save_file)
        del save_generator

    def random_flip_inputs(self, inputs):
        flip = (random.random() <= self.flip_rate)
        if flip:
            inputs = torch.flip(inputs, dims=(3,))
        return inputs


def class_balance_loss(output):
    logit_mu = output.softmax(dim=1).mean(dim=0)
    num_classes = output.shape[1]
    # ignore sign
    entropy = (logit_mu * logit_mu.log() / math.log(num_classes)).sum()
    return 1 + entropy
