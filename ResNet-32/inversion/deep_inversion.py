# -*-coding:utf8-*-

import os
import torch
import torch.nn.functional as F
import pickle
import copy
import random

from inversion import functions
from inversion import inversion_scheduler


class DeepInversionRunner(object):
    def __init__(self, model, input_shape, max_iter, lr, alpha_tv, alpha_l2, alpha_rf, alpha_adi, num_class,
                 use_cuda, adi_iters=0, scheduler_params=None, local_path=None, flip_input=False, log_step=1000):
        self.local_path = local_path
        if self.local_path is not None and (not os.path.exists(self.local_path)):
            os.makedirs(self.local_path)
        self.model = model
        self.model.eval()
        self.input_shape = input_shape
        self.max_iter = max_iter
        self.lr = lr
        self.alpha_tv = alpha_tv
        self.alpha_l2 = alpha_l2
        self.alpha_rf = alpha_rf
        self.alpha_adi = alpha_adi
        self.num_class = num_class
        self.use_cuda = use_cuda
        self.adi_iters = adi_iters
        self.scheduler_params = scheduler_params
        self.local_path = None
        self.flip_input = flip_input
        self.log_step = log_step
        # add hooks
        self.stat_hooks = []
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    def register_hooks(self):
        for n, mi in self.model.named_modules():
            if isinstance(mi, torch.nn.BatchNorm2d):
                self.stat_hooks.append(functions.L2BNOutputHook(module=mi))

    def remove_hooks(self):
        for hi in self.stat_hooks:
            hi.remove_hook()
        self.stat_hooks.clear()

    def adaptive_inversion_loss(self, inputs, outputs, student_model):
        t = 3.0
        out_stu = student_model(inputs)
        p = F.softmax(out_stu / t, dim=1)
        if outputs is None:
            out_tch = self.model(inputs)
        else:
            out_tch = outputs
        q = F.softmax(out_tch / t, dim=1)
        m = 0.5 * (p + q)

        p = torch.clamp(p, 0.01, 0.99)
        q = torch.clamp(q, 0.01, 0.99)
        m = torch.clamp(m, 0.01, 0.99)
        eps = 0.0
        loss_verifier_cig = 0.5 * self.kl_loss(torch.log(p + eps), m) + 0.5 * self.kl_loss(torch.log(q + eps), m)
        # JS criteria - 0 means full correlation, 1 - means completely different
        loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
        return loss_verifier_cig

    def inversion(self, inputs, target, opt, scheduler, iters=None, return_best=False, student_model=None):
        losses = {}
        best_loss = None
        best_inputs = None
        if iters is None:
            iters = self.max_iter
        for i in range(iters):
            if scheduler is not None:
                scheduler.step(i)
            inputs_jit = augment(inputs)
            if self.flip_input:
                flip_inputs = random_flip_inputs(inputs=inputs_jit)
            else:
                flip_inputs = inputs_jit
            output = self.model(flip_inputs)
            l_ce = self.ce_loss(output, target)
            l_stat = torch.stack([h.l2_stats_regularization() for h in self.stat_hooks]).sum()
            l_l2 = torch.norm(inputs_jit, 2)
            l_tv_l1, l_tv_l2 = total_variance(inputs=inputs_jit)
            if student_model is not None:
                l_adi = self.adaptive_inversion_loss(inputs=inputs_jit, outputs=output, student_model=student_model)
            else:
                l_adi = 0
            loss = l_ce + self.alpha_tv * l_tv_l2 + self.alpha_l2 * l_l2 + self.alpha_rf * l_stat \
                + self.alpha_adi * l_adi
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % self.log_step == 0:
                print('finish training step:', i)
                if student_model is not None:
                    print('ce loss:', l_ce.item(), 'l2 loss:', l_l2.item(), 'stat loss:', l_stat.item(), 'total var:',
                          l_tv_l2.item(), 'adi loss:', l_adi.item(), 'total loss:', loss.item())
                else:
                    print('ce loss:', l_ce.item(), 'l2 loss:', l_l2.item(), 'stat loss:', l_stat.item(), 'total var:',
                          l_tv_l2.item(), 'total loss:', loss.item())
            if i % 10 == 0:
                if student_model is not None:
                    losses[i] = {'ce loss': l_ce.item(), 'l2 loss': l_l2.item(), 'stat loss': l_stat.item(),
                                 'total var': l_tv_l2.item(), 'adi loss': l_adi.item(), 'total loss': loss.item()}
                else:
                    losses[i] = {'ce loss': l_ce.item(), 'l2 loss': l_l2.item(), 'stat loss': l_stat.item(),
                                 'total var': l_tv_l2.item(), 'total loss': loss.item()}
            if best_loss is None or best_loss > loss.item():
                best_loss = loss.item()
                best_inputs = copy.deepcopy(inputs.clone().detach())
        if self.local_path is not None:  # dump loss during training
            classes = self.num_class
            loss_file = os.path.join(self.local_path, 'inv_train_logs_' + str(classes) + '.pkl')
            with open(loss_file, 'wb') as fw:
                pickle.dump(losses, fw)
        if return_best:
            return best_inputs
        else:
            return inputs

    def get_data(self, batch_size, lab=None, return_best=False, student_model=None):
        if student_model is not None:
            student_model.eval()
        self.register_hooks()
        if lab is None:
            target = torch.randint(self.num_class, (batch_size,)).long()
        else:
            target = torch.ones([batch_size, ], dtype=torch.long) * lab
        shape = [target.shape[0], ] + self.input_shape
        if self.use_cuda:
            inputs = torch.randn(shape, device='cuda').requires_grad_(True)
            target = target.cuda()
        else:
            inputs = torch.randn(shape, device='cpu').requires_grad_(True)
        opt = torch.optim.Adam([inputs], lr=self.lr)
        if self.scheduler_params is not None:
            scheduler = inversion_scheduler.CosineLRScheduler(
                optimizer=opt, warmup=self.scheduler_params['warmup'], total_epoch=self.max_iter, base_lr=self.lr)
        else:
            scheduler = None
        inputs = self.inversion(
            inputs=inputs, target=target, opt=opt, scheduler=scheduler,
            return_best=return_best, student_model=student_model
        )
        self.remove_hooks()
        if student_model is not None:
            student_model.train()
        return inputs.clone().detach(), target

    def adapt_data(self, student_model, inputs, lab, return_best=False):
        """
        Used for modify existing samples with adaptive inversion
        :param student_model:
        :param inputs:
        :param lab:
        :param return_best:
        :return:
        """
        student_model.eval()
        self.register_hooks()
        opt = torch.optim.Adam([inputs], lr=self.lr)
        if self.scheduler_params is not None:
            scheduler = inversion_scheduler.CosineLRScheduler(
                optimizer=opt, warmup=self.scheduler_params['warmup'], total_epoch=self.adi_iters, base_lr=self.lr)
        else:
            scheduler = None
        inputs = self.inversion(
            inputs=inputs, target=lab, opt=opt, scheduler=scheduler, iters=self.adi_iters,
            return_best=return_best, student_model=student_model
        )
        self.remove_hooks()
        student_model.train()
        return inputs.clone().detach()

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def save_models(self):
        if self.local_path is None:
            return
        classes = self.num_class
        save_model = copy.deepcopy(self.model).cpu().eval()
        save_file = os.path.join(self.local_path, 'inv_model_' + str(classes) + '.pkl')
        torch.save(save_model, save_file)
        del save_model


def total_variance(inputs):
    diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
    diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
    diff3 = inputs[:, :, 1:, :-1] - inputs[:, :, :-1, 1:]
    diff4 = inputs[:, :, :-1, :-1] - inputs[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def augment(inputs):
    lim_0, lim_1 = 2, 2
    # apply random jitter offsets
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    input_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))
    return input_jit


def random_flip_inputs(inputs):
    flip = random.random() > 0.5
    if flip:
        inputs = torch.flip(inputs, dims=(3, ))
    return inputs
