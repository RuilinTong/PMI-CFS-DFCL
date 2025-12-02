# -*-coding:utf8-*-
# Class-wise inversion without using generator

import os
import torch
import torch.nn.functional as F
import random
import copy
import pickle
import numpy as np

from inversion import functions
from inversion import inversion_scheduler
from backbone import resnet_cifar


class ClassWiseContrastiveInversion(object):
    def __init__(self, local_path, model, image_size, lr, train_steps, alpha_pr, alpha_rf, scheduler_params,
                 hook_type='l2', hook_rate=1.0, mse_type='l2', flip_rate=0, log_step=1000, opt_type='adam',
                 boost_factor=True):
        self.local_path = local_path
        if self.local_path is not None and not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.model = model
        self.image_size = image_size
        self.lr = lr
        self.train_steps = train_steps
        self.alpha_pr = alpha_pr
        self.alpha_rf = alpha_rf
        self.cls2mean = []
        self.cls2std = []
        self.scheduler_params = scheduler_params
        self.hook_type = hook_type
        self.hook_rate = hook_rate
        self.flip_rate = flip_rate
        self.log_step = log_step
        self.opt_type = opt_type
        self.boost_factor = boost_factor
        # hooks
        self.on_cuda = False
        self.stat_hooks = []
        self.gaussian_kernel = functions.Gaussiansmoothing(channels=3, kernel_size=5, sigma=1)
        self.mse_type = mse_type
        if self.mse_type == 'l1':
            self.mse_loss = torch.nn.L1Loss()
        else:
            self.mse_loss = torch.nn.MSELoss()
        self.pool_layer = torch.nn.AdaptiveAvgPool2d(1)
        self.best_layer_lrs = []
        self.best_layer_rfs = []

    def update_model(self, model):
        # if self.on_cuda:
        #     self.model.cpu()
        self.model = model
        if self.on_cuda:
            self.model.cuda()
        # need to tune parameters with model is updated
        self.best_layer_lrs.clear()
        self.best_layer_rfs.clear()

    def register_hooks(self, model=None):
        if model is None:
            model = self.model
            flg_none = True
        else:
            flg_none = False
        all_name = []
        for n, mi in model.named_modules():
            if isinstance(mi, torch.nn.BatchNorm2d):
                all_name.append(n)
        # sorted_names = sorted(all_name)
        hook_num = max(int(len(all_name) * self.hook_rate), 1)
        name_pool = []
        for i in range(hook_num):
            name_pool.append(all_name[i])
        if self.hook_type == 'l2':
            for n, mi in model.named_modules():
                if isinstance(mi, torch.nn.BatchNorm2d) and n in name_pool:
                    self.stat_hooks.append(functions.L2BNOutputHook(module=mi))
        elif self.hook_type == 'kl':
            for n, mi in model.named_modules():
                if isinstance(mi, torch.nn.BatchNorm2d) and n in name_pool:
                    self.stat_hooks.append(functions.ModifiedBNInputHook(module=mi))
        elif self.hook_type == 'cos_kl' and flg_none:
            stat_file = os.path.join(self.local_path, 'bn_stats.pkl')
            if not os.path.exists(stat_file):
                raise ValueError('BN stat file does not exist')
            with open(stat_file, 'rb') as fr:
                name2stat = pickle.load(fr)
            for n, mi in self.model.named_modules():
                if isinstance(mi, torch.nn.BatchNorm2d) and n in name_pool:
                    self.stat_hooks.append(
                        functions.CustomBNInputHook(
                            module=mi,
                            running_mean=name2stat[n][0].cuda() if self.on_cuda else name2stat[n][0].cuda(),
                            running_var=name2stat[n][1].cuda() if self.on_cuda else name2stat[n][1].cuda()
                        )
                    )
        else:
            raise ValueError('Invalid hook type')

    def remove_hooks(self):
        for hi in self.stat_hooks:
            hi.remove_hook()
        self.stat_hooks.clear()

    def criterion_pr(self, inputs):
        input_pad = F.pad(inputs, (2, 2, 2, 2), mode="reflect")
        input_smooth = self.gaussian_kernel(input_pad).detach()
        return F.mse_loss(inputs, input_smooth)

    def inversion(self, inputs, target_feats, opt, scheduler, model=None, iters=None, return_best=False,
                  use_pool=False, input_loss=None, alpha_feat=None, verbose=True):
        # losses = {}
        best_loss = None
        best_inputs = None
        if iters is None:
            iters = self.train_steps
        if alpha_feat is None:
            alpha_rf = self.alpha_rf
            alpha_in = self.alpha_rf
        else:
            alpha_rf = alpha_feat
            alpha_in = alpha_feat
        alpha_mse = 1.0
        init_losses = {}
        for i in range(iters):
            if model is None:
                out_feat = self.pool_layer(self.model.backbone(inputs))[:, :, 0, 0]
            else:
                out_feat = model(inputs)
                if use_pool:
                    out_feat = self.pool_layer(out_feat)[:, :, 0, 0]
            if self.hook_type == 'l2':
                l_stat = torch.stack([h.l2_stats_regularization() for h in self.stat_hooks]).sum()
            elif self.hook_type == 'kl' or self.hook_type == 'cos_kl':
                l_stat = torch.stack([h.stats_regularization() for h in self.stat_hooks]).sum()
            else:
                l_stat = 0
            # l_blur = self.criterion_pr(inputs=inputs)
            l_mse = self.mse_loss(out_feat, target_feats)
            # loss = l_mse + self.alpha_pr * l_blur + self.alpha_rf * l_stat
            if i == 0:
                init_losses['mse'] = l_mse.item()
                init_losses['stat'] = l_stat.item()
            else:  # prevent non-convergence
                if l_mse.item() > init_losses['mse'] and self.boost_factor:
                    alpha_mse = min(alpha_mse * 2, 10)
                if l_stat.item() > init_losses['stat'] and self.boost_factor:
                    alpha_rf = min(alpha_rf * 2, 10 * self.alpha_rf)
            loss = alpha_mse * l_mse + alpha_rf * l_stat
            if input_loss is not None:
                l_in = input_loss.compute_loss(inputs)
                if i == 0:
                    init_losses['in'] = l_in.item()
                else:
                    if l_in.item() > init_losses['in'] and self.boost_factor:
                        alpha_in = min(alpha_in * 2, 5 * self.alpha_rf)
                loss = loss + alpha_in * l_in
            else:
                l_in = None
            opt.zero_grad()
            loss.backward()
            opt.step()
            cmp_loss = l_mse.item() + self.alpha_rf * l_stat.item()
            if l_in is not None:
                cmp_loss += self.alpha_rf * l_in.item()
            if (i % self.log_step == 0 or i == iters - 1) and verbose:
                print('finish training step:', i)
                print('mse loss:', l_mse.item(), 'smooth loss:', 'none', 'stat loss:', l_stat.item(),
                      'total loss:', cmp_loss, end=' ' if l_in is not None else '\n')
                if l_in is not None:
                    print('input loss:', l_in.item())
            if best_loss is None or best_loss > cmp_loss:
                best_loss = copy.deepcopy(float(cmp_loss))
                best_inputs = copy.deepcopy(inputs.clone().detach())
            if scheduler is not None:  # fix the problem of scheduler called before optimizer
                scheduler.step(i)
        if return_best:
            return best_inputs, best_loss
        else:
            return inputs, best_loss

    def get_data(self, target_feats, iters=None, return_best=False, init_input=None, relabel_input=False):
        self.model.eval()
        self.register_hooks()
        shape = [target_feats.shape[0], ] + self.image_size
        if self.on_cuda:
            if init_input is None:
                inputs = torch.randn(shape, device='cuda').requires_grad_(True)
            else:  # option for initializing inputs
                inputs = torch.tensor(init_input, device='cuda', requires_grad=True)
        else:
            if init_input is None:
                inputs = torch.randn(shape, device='cpu').requires_grad_(True)
            else:  # option for initializing inputs
                inputs = torch.tensor(init_input, device='cpu', requires_grad=True)
        if self.opt_type == 'adam':
            opt = torch.optim.Adam([inputs], lr=self.lr)
        else:
            opt = torch.optim.SGD([inputs], lr=self.lr)
        if self.scheduler_params is not None:
            scheduler = inversion_scheduler.CosineLRScheduler(
                optimizer=opt, warmup=self.scheduler_params['warmup'], total_epoch=self.train_steps, base_lr=self.lr)
        else:
            scheduler = None
        inputs, best_loss = self.inversion(
            inputs=inputs, target_feats=target_feats, opt=opt, scheduler=scheduler,
            return_best=return_best, iters=iters
        )
        self.remove_hooks()
        inputs.clone().detach()
        if relabel_input:
            with torch.no_grad():
                out_logit = self.model(inputs)
                target = out_logit.argmax(dim=1)
        return inputs, target

    def layer_wise_inversion_for_cl(self, batch_size, target_feats, target, finetune_iters, finetune_lr, iters=None,
                                    return_best=True, rf_factor=1.0, search_param=True, selection_agent=None):
        """
        layer-wise inversion, do model inversion per layer, aims to accelerate model inversion.
        :param batch_size:
        :param target_feats:
        :param target
        :param finetune_iters:
        :param finetune_lr:
        :param iters:
        :param return_best:
        :param rf_factor:
        :param search_param:
        :param selection_agent:
        :return:
        """
        self.model.eval()
        if self.on_cuda:
            target_feats = target_feats.cuda()
            target = target.cuda()
        layer_batch = target_feats.shape[0]
        # get all blocks
        all_blocks = resnet_cifar.get_all_blocks(resnet_model=self.model.backbone)
        print('total blocks:', len(all_blocks))
        # load block input stats
        block_input_file = os.path.join(self.local_path, 'block_input_stat.pkl')
        if os.path.exists(block_input_file):
            with open(block_input_file, 'rb') as fr:
                bid2stat = pickle.load(fr)
        else:
            bid2stat = None
        out_feats = target_feats
        # --- layer wise inversion ---
        shape = [target_feats.shape[0], ] + self.image_size
        out_shape = target_feats.shape
        opt_inputs = None
        # for different loss factor in layer-wise inversion
        self.alpha_rf = self.alpha_rf * len(all_blocks) * rf_factor
        candidate_lrs = [self.lr * 2, self.lr, self.lr * 0.5, self.lr * 0.25, self.lr * 0.125]
        candidate_rfs = [self.alpha_rf, self.alpha_rf * 0.5, self.alpha_rf * 0.25, self.alpha_rf * 0.125]
        if not search_param:
            self.best_layer_lrs = [self.lr] * len(all_blocks)
            self.best_layer_rfs = [self.alpha_rf] * len(all_blocks)
        for i in range(len(all_blocks) - 1, -1, -1):
            # get input and output shape
            input_shape, expect_out_shape = self.get_module_input_shape(
                module=all_blocks[i], all_blocks=all_blocks, shape=shape,
                pool_layer=None if i < len(all_blocks) - 1 else self.pool_layer
            )
            if out_shape != expect_out_shape:
                raise ValueError('Output shapes do not match:', out_shape, expect_out_shape)
            # search best learning rate for each layer at the first time
            # different layers do have different best learning rates
            if len(self.best_layer_lrs) == len(all_blocks):
                lrs = [self.best_layer_lrs[i]]
            else:
                lrs = candidate_lrs
            # test function: search best alpha-rf for inversion.
            if len(self.best_layer_rfs) == len(all_blocks):
                alpha_rfs = [self.best_layer_rfs[i]]
            else:
                alpha_rfs = candidate_rfs
            best_cand_loss = None
            best_lr = None
            best_alpha_rf = None
            best_inputs = None
            for lr in lrs:
                for alpha_rf in alpha_rfs:
                    # initialize input
                    if self.on_cuda:
                        if i == 0:
                            inputs = torch.randn(input_shape, device='cuda').requires_grad_(True)
                            input_loss = None
                        else:
                            inputs, input_loss = self.init_layer_input(
                                all_blocks=all_blocks, idx=i, input_shape=input_shape, bid2stat=bid2stat)
                    else:
                        if i == 0:
                            inputs = torch.randn(input_shape, device='cpu').requires_grad_(True)
                            input_loss = None
                        else:
                            inputs, input_loss = self.init_layer_input(
                                all_blocks=all_blocks, idx=i, input_shape=input_shape, bid2stat=bid2stat)
                    # build optimizer
                    if self.opt_type == 'adam':
                        opt = torch.optim.Adam([inputs], lr=lr)
                    else:
                        opt = torch.optim.SGD([inputs], lr=lr)
                    # add option for add scheduler
                    if 'layer_schedule' in self.scheduler_params and self.scheduler_params['layer_schedule']:
                        layer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer=opt,
                            milestones=[int(self.train_steps * 0.5), int(self.train_steps * 0.75)],
                            gamma=self.scheduler_params['lr_rate']
                        )
                    else:
                        layer_scheduler = None
                    self.register_hooks(model=all_blocks[i])
                    # model inversion training
                    opt_inputs, best_loss = self.inversion(
                        inputs=inputs,
                        target_feats=out_feats,
                        opt=opt,
                        scheduler=layer_scheduler,
                        model=all_blocks[i],
                        iters=iters,
                        return_best=return_best,
                        use_pool=False if i < len(all_blocks) - 1 else True,
                        input_loss=input_loss,
                        alpha_feat=alpha_rf,
                        verbose=False
                    )
                    self.remove_hooks()
                    best_loss_cross = 0
                    if best_cand_loss is None or best_cand_loss > (best_loss + best_loss_cross):
                        best_cand_loss = (best_loss + best_loss_cross)
                        best_lr = lr
                        best_alpha_rf = alpha_rf
                        best_inputs = opt_inputs.clone().detach()  # fix error of best inputs
                    del opt
            if len(self.best_layer_lrs) < len(all_blocks) or len(self.best_layer_rfs) < len(all_blocks):
                print('best learning rate and alpha rf for block', i, 'is:', best_lr, best_alpha_rf)
                self.best_layer_lrs = [best_lr] + self.best_layer_lrs
                self.best_layer_rfs = [best_alpha_rf] + self.best_layer_rfs
            opt_inputs = best_inputs
            # update output
            out_feats = opt_inputs.clone().detach()
            out_shape = opt_inputs.shape
            # all_inter_inputs.append(opt_inputs.cpu())
        self.alpha_rf = self.alpha_rf / (len(all_blocks) * rf_factor)
        # finetune inputs with the full model
        print('model inversion on full model')
        init_input = opt_inputs.cpu().numpy()
        tuned_inputs = []
        steps = int(layer_batch // batch_size)
        if layer_batch % batch_size != 0:
            steps += 1
        for i in range(steps):
            start = i * batch_size
            end = min((i + 1) * batch_size, layer_batch)
            inputs = torch.tensor(init_input[start:end, :], device='cuda', requires_grad=True)
            if self.opt_type == 'adam':
                opt = torch.optim.Adam([inputs], lr=finetune_lr)
            else:
                opt = torch.optim.SGD([inputs], lr=finetune_lr)
            step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=opt,
                milestones=self.scheduler_params['milestones'],
                gamma=self.scheduler_params['lr_rate']
            )
            scheduler = WarmupScheduler(
                optimizer=opt, scheduler=step_scheduler, warmup=self.scheduler_params['warmup'], lr=finetune_lr)
            self.register_hooks(model=None)
            opt_inputs, best_loss = self.inversion(
                inputs=inputs, target_feats=target_feats[start:end, :], opt=opt, scheduler=scheduler,
                return_best=return_best, iters=finetune_iters, model=None
            )
            self.remove_hooks()
            print('best loss after tuning is:', best_loss)
            opt_inputs.clone().detach()
            tuned_inputs.append(opt_inputs)
        tuned_inputs = torch.cat(tuned_inputs, dim=0)
        slt_ind = np.arange(tuned_inputs.shape[0])
        return tuned_inputs, slt_ind

    def get_module_input_shape(self, all_blocks, module, shape, pool_layer=None):
        model = torch.nn.Sequential(*all_blocks)
        inputs = torch.randn(shape, requires_grad=False)
        if self.on_cuda:
            inputs = inputs.cuda()
        hook = InputShapeHook(module=module, pool_layer=pool_layer)
        with torch.no_grad():
            _ = model(inputs)
        input_shape, output_shape = hook.get_input_output_shape()
        hook.remove_hook()
        return input_shape, output_shape

    def init_layer_input(self, all_blocks, idx, input_shape, bid2stat=None):
        # The output of block is residual sum, better to compute the distribution of input
        if bid2stat is None:
            prv_block = all_blocks[idx - 1]
            last_bn = None
            for n, mi in prv_block.named_modules():
                if isinstance(mi, torch.nn.BatchNorm2d):
                    last_bn = mi
            if last_bn is None:
                return None
            weight = last_bn.weight.detach().cpu().numpy()
            bias = last_bn.bias.detach().cpu().numpy()
            weight = np.expand_dims(weight, axis=[0, 2, 3])
            bias = np.expand_dims(bias, axis=[0, 2, 3])
        else:
            bias, var = bid2stat[idx]
            bias = bias.cpu().numpy()
            var = var.cpu().numpy()
            weight = np.sqrt(var)
            weight = np.expand_dims(weight, axis=[0, 2, 3])
            bias = np.expand_dims(bias, axis=[0, 2, 3])
        init_value = torch.randn(input_shape, device='cpu').numpy()
        init_input = init_value * weight + bias
        input_loss = KLDivergence(
            mean=torch.tensor(bias, requires_grad=False), var=torch.tensor(weight ** 2, requires_grad=False))
        if self.on_cuda:
            inputs = torch.tensor(init_input, device='cuda', requires_grad=True)
            input_loss.cuda()
        else:
            inputs = torch.tensor(init_input, device='cpu', requires_grad=True)
            input_loss.cpu()
        return inputs, input_loss

    def random_flip_inputs(self, inputs):
        flip = (random.random() <= self.flip_rate)
        if flip:
            inputs = torch.flip(inputs, dims=(3,))
        return inputs

    def get_sample_loss(self, samples, labels):
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        steps = int(samples.shape[0] / 100)
        if samples.shape[0] % 100 != 0:
            steps += 1
        all_loss = []
        with torch.no_grad():
            for i in range(steps):
                start = 100 * i
                end = min(samples.shape[0], (i + 1) * 100)
                out_logit = self.model(samples[start:end, :]).detach()
                loss = loss_fn(out_logit, labels[start:end]).cpu().numpy()
                for j in range(loss.shape[0]):
                    all_loss.append(loss[j])
        return all_loss

    def cuda(self):
        self.model.cuda()
        self.on_cuda = True

    def cpu(self):
        self.model.cpu()
        self.on_cuda = False


class InputShapeHook(object):
    def __init__(self, module, pool_layer=None):
        self.module = module
        self.inputs = None
        self.outputs = None
        self.handle = self.module.register_forward_hook(hook=self.get_input_hook())
        self.pool_layer = pool_layer

    def get_input_output_shape(self):
        if isinstance(self.outputs, tuple) or isinstance(self.outputs, list):  # for stage computation
            out_shape = self.outputs[1].shape
        else:
            if self.pool_layer is not None:
                temp_out = self.pool_layer(self.outputs)[:, :, 0, 0]
                out_shape = temp_out.shape
            else:
                out_shape = self.outputs.shape
        return self.inputs.shape, out_shape

    def get_input_hook(self):

        def hook(module, input, output):
            self.inputs = input[0]
            self.outputs = output

        return hook

    def remove_hook(self):
        self.inputs = None
        self.module = None
        self.handle.remove()


class KLDivergence(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def compute_loss(self, x):
        running_mean = self.mean
        running_var = self.var
        # get stats of input
        mean = x.mean([0, 2, 3])
        nch = x.shape[1]
        value = x.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8
        # compute KL divergence
        r = torch.log(running_var) - torch.log(var + 1e-8) - (1 - (var + (mean - running_mean) ** 2) / running_var)
        r = r.mean() * 0.5
        return r

    def cuda(self):
        self.mean = self.mean.cuda()
        self.var = self.var.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.var = self.var.cpu()


class WarmupScheduler(object):
    def __init__(self, optimizer, scheduler, warmup, lr):
        self.warmup = warmup
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.epoch = 0
        # initialize learning rate
        temp_lr = self.lr * (self.epoch + 1) / max(self.warmup, 1)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = temp_lr

    def step(self, i=-1):
        if self.epoch < self.warmup:
            lr = self.lr * (self.epoch + 1) / self.warmup
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
        self.epoch += 1


def match_bins(losses, bin2bound):
    id2bin = []
    for i in range(len(losses)):
        li = losses[i]
        for j in range(len(bin2bound)):
            low, up = bin2bound[j]
            if low is None:
                if li <= up:
                    id2bin.append(j)
                    break
                else:
                    continue
            if up is None:
                if li > low:
                    id2bin.append(j)
                    break
            if low < li <= up:
                id2bin.append(j)
                break
    return id2bin
