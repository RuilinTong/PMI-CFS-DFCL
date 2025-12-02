# -*-coding:utf8-*-

import torch
import numpy as np
import random


def get_class_wise_distribution(backbone, data_loader, classes, on_cuda):
    train_stat = backbone.training
    backbone.eval()
    cls2feats = {}
    for ci in classes:
        cls2feats[ci] = []
    pool_layer = torch.nn.AdaptiveAvgPool2d(1)
    for data in data_loader:
        sp, lab = data
        if on_cuda:
            sp = sp.cuda()
        out_feat = pool_layer(backbone(sp))[:, :, 0, 0]
        out_feat = out_feat.detach()
        if on_cuda:
            out_feat = out_feat.cpu()
        out_feat = out_feat.numpy()
        for i in range(sp.shape[0]):
            cls = int(lab[i])
            cls2feats[cls].append(out_feat[i, :])
    cls2mean = {}
    cls2std = {}
    for ci in cls2feats.keys():
        all_feats = np.stack(cls2feats[ci], axis=0)
        cls2mean[ci] = np.mean(all_feats, axis=0)
        cls2std[ci] = np.std(all_feats, axis=0)
    backbone.train(train_stat)
    return cls2mean, cls2std


def get_class_wise_distribution_generator(backbone, generator, num_classes, class_num, on_cuda):
    train_stat = backbone.training
    backbone.eval()
    classes = list(range(num_classes))
    cls2feats = {}
    for ci in classes:
        cls2feats[ci] = []
    pool_layer = torch.nn.AdaptiveAvgPool2d(1)
    finish_class = set()
    step = 0
    while len(finish_class) < num_classes:
        inv_data = generator.sample(batch_size=128)
        if len(inv_data) == 2:
            inv_sp, inv_lab = inv_data
        else:
            inv_sp, inv_lab, _ = inv_data
        out_feat = pool_layer(backbone(inv_sp))[:, :, 0, 0]
        out_feat = out_feat.detach()
        if on_cuda:
            out_feat = out_feat.cpu().numpy()
            inv_lab = inv_lab.cpu().numpy()
        for i in range(inv_sp.shape[0]):
            ci = int(inv_lab[i])
            if ci in finish_class:
                continue
            cls2feats[ci].append(out_feat[i, :])
            if len(cls2feats[ci]) == class_num:
                finish_class.add(ci)
        step += 1
        if step == 10000:
            break
    cls2mean = {}
    cls2std = {}
    for ci in cls2feats.keys():
        all_feats = np.stack(cls2feats[ci], axis=0)
        cls2mean[ci] = np.mean(all_feats, axis=0)
        cls2std[ci] = np.mean(all_feats, axis=0)
    backbone.train(train_stat)
    return cls2mean, cls2std


def get_losses(data_loader, model, on_cuda):
    train_stat = model.training
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    losses = []
    with torch.no_grad():
        for data in data_loader:
            sp, lab = data
            if on_cuda:
                sp = sp.cuda()
                lab = lab.cuda()
            out_logit = model(sp)
            loss = loss_fn(out_logit, lab)
            if on_cuda:
                loss = loss.cpu()
            loss = loss.numpy()
            for i in range(sp.shape[0]):
                losses.append(loss[i])
    model.train(train_stat)
    return losses


def get_loss_from_dist(cls2mean, cls2std, head, total_num, on_cuda):
    train_stat = head.training
    head.eval()
    feat_dim = cls2mean[0].shape[0]
    cls_means = []
    cls_stds = []
    for ci in range(head.num_classes):
        cls_means.append(cls2mean[ci])
        cls_stds.append(cls2std[ci])
    cls_means = torch.tensor(np.stack(cls_means, axis=0), dtype=torch.float32, requires_grad=False)
    cls_stds = torch.tensor(np.stack(cls_stds, axis=0), dtype=torch.float32, requires_grad=False)
    steps = int(total_num // 200)
    if total_num % 200 != 0:
        steps += 1
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    losses = []
    with torch.no_grad():
        for i in range(steps):
            batch_size = min(200, total_num - len(losses))
            target = torch.tensor(np.random.randint(
                size=[batch_size], low=0, high=head.num_classes), dtype=torch.long, requires_grad=False)
            rand_feats = torch.randn([target.shape[0], feat_dim], dtype=torch.float32, requires_grad=False)
            means = cls_means[target, :]
            stds = cls_stds[target, :]
            target_feats = rand_feats * stds + means
            target_feats = target_feats.detach()
            if on_cuda:
                target = target.cuda()
                target_feats = target_feats.cuda()
            out_logit = head.classify(target_feats)
            loss = loss_fn(out_logit, target)
            if on_cuda:
                loss = loss.cpu()
            loss = loss.numpy()
            for j in range(target_feats.shape[0]):
                losses.append(loss[j])
    head.train(train_stat)
    return losses


def get_inv_losses(samples, labs, model, on_cuda):
    train_stat = model.training
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    losses = []
    steps = int(samples.shape[0] // 200)
    if samples.shape[0] % 200 != 0:
        steps += 1
    with torch.no_grad():
        for si in range(steps):
            start = si * 200
            end = min((si + 1) * 200, samples.shape[0])
            sp = samples[start:end, :]
            lab = labs[start:end]
            if on_cuda:
                sp = sp.cuda()
                lab = lab.cuda()
            out_logit = model(sp)
            loss = loss_fn(out_logit, lab)
            if on_cuda:
                loss = loss.cpu()
            loss = loss.numpy()
            for i in range(sp.shape[0]):
                losses.append(loss[i])
    model.train(train_stat)
    return losses


def estimate_class_stats(ori_model, tgt_model, cls2sp, ori_cls2mean, ori_cls2std, use_cuda):
    """
    used for estimating class2mean and class2std with only few data
    :param ori_model: model trained after of last task
    :param tgt_model: model trained after current task
    :param cls2sp: dict, map class to list of samples, each sample is torch.Tensor
    :param ori_cls2mean: class2mean of ori-model.
    :param ori_cls2std: class2std of ori-model.
    :param use_cuda:
    :return:
    """
    ori_train_stat = ori_model.training
    tgt_train_stat = tgt_model.training
    ori_model.eval()
    tgt_model.eval()
    # compute features of each sample
    pool_layer = torch.nn.AdaptiveAvgPool2d(1)
    ori_cls2feat = {}
    tgt_cls2feat = {}
    for ci in cls2sp.keys():
        c_sps = torch.stack(cls2sp[ci], dim=0)
        if use_cuda:
            c_sps = c_sps.cuda()
        ori_out_feat = pool_layer(ori_model.backbone(c_sps))[:, :, 0, 0].detach()
        tgt_out_feat = pool_layer(tgt_model.backbone(c_sps))[:, :, 0, 0].detach()
        if use_cuda:
            ori_out_feat = ori_out_feat.cpu()
            tgt_out_feat = tgt_out_feat.cpu()
        ori_cls2feat[ci] = ori_out_feat
        tgt_cls2feat[ci] = tgt_out_feat
    # compute mean shift
    cls2mean_shift = {}
    for ci in cls2sp.keys():
        feat_shift = tgt_cls2feat[ci] - ori_cls2feat[ci]
        mean_shift = torch.mean(feat_shift, dim=0)
        cls2mean_shift[ci] = mean_shift.cpu().numpy()
    # compute distance change
    cls2dist_shift = {}
    max_pair = 10000
    max_sub_pair = 1000
    for ci in cls2sp.keys():
        all_pairs = []
        for i in range(len(cls2sp[ci]) - 1):
            for j in range(i + 1, len(cls2sp[ci])):
                all_pairs.append([i, j])
            if len(all_pairs) == max_pair:
                break
        if len(all_pairs) > max_sub_pair:
            pairs = random.sample(all_pairs, max_sub_pair)
        else:
            pairs = all_pairs
        total_log_scale = None
        ori_feats = ori_cls2feat[ci]
        tgt_feats = tgt_cls2feat[ci]
        for pi in pairs:
            ori_dist = (ori_feats[pi[0]] - ori_feats[pi[1]]) ** 2
            tgt_dist = (tgt_feats[pi[0]] - tgt_feats[pi[1]]) ** 2
            scale = (tgt_dist + 1e-6) / (ori_dist + 1e-6)
            log_scale = torch.log(scale) / 2   # do sqrt
            if total_log_scale is None:
                total_log_scale = log_scale
            else:
                total_log_scale = total_log_scale + log_scale
        avg_log_scale = total_log_scale / len(pairs)
        avg_scale = torch.exp(avg_log_scale)
        cls2dist_shift[ci] = avg_scale.cpu().numpy()
    ori_model.train(ori_train_stat)
    tgt_model.train(tgt_train_stat)
    tgt_cls2mean = {}
    tgt_cls2std = {}
    for ci in cls2sp.keys():
        tgt_mean = ori_cls2mean[ci] + cls2mean_shift[ci]
        tgt_std = ori_cls2std[ci] * cls2dist_shift[ci]
        tgt_cls2mean[ci] = tgt_mean
        tgt_cls2std[ci] = tgt_std
    return tgt_cls2mean, tgt_cls2std


def get_bn_input_stat(data_loader, model, use_cuda):
    train_stat = model.training
    model.eval()
    # register hooks
    stat_hooks = []
    for n, mi in model.named_modules():
        if isinstance(mi, torch.nn.BatchNorm2d):
            stat_hooks.append(BNInputStatHook(module=mi, module_name=n))
    # forward computation
    with torch.no_grad():
        for data in data_loader:
            sp, lab = data
            if use_cuda:
                sp = sp.cuda()
            _ = model(sp)
            for hi in stat_hooks:
                hi.update_stat()
    # get stat
    name2stat = {}
    for hi in stat_hooks:
        name = hi.get_module_name()
        mean, var = hi.compute_stat()
        name2stat[name] = [mean, var]
    model.train(train_stat)
    return name2stat


class BNInputStatHook(object):
    def __init__(self, module, module_name):
        self.module = module
        self.module_name = module_name
        self.inputs = None
        self.handle = self.module.register_forward_hook(hook=self.get_input_hook())
        self.cnt = 0
        self.mean = 0
        self.mean_square = 0
        self.var = 0

    def update_stat(self):
        # get stats of input
        batch_size = self.inputs.shape[0]
        mean = self.inputs.mean([0, 2, 3]).detach().cpu()
        mean_square = (self.inputs ** 2).mean([0, 2, 3]).detach().cpu()
        self.mean = self.cnt / (self.cnt + batch_size) * self.mean + batch_size / (self.cnt + batch_size) * mean
        self.mean_square = self.cnt / (self.cnt + batch_size) * self.mean_square + \
            batch_size / (self.cnt + batch_size) * mean_square
        self.cnt += batch_size

    def compute_stat(self):
        self.var = self.mean_square - self.mean ** 2
        return self.mean, self.var

    def get_module_name(self):
        return self.module_name

    def get_input_hook(self):

        def hook(module, input, output):
            self.inputs = input[0]

        return hook

    def remove_hook(self):
        self.inputs = None
        self.module = None
        self.handle.remove()
