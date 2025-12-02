# -*-coding:utf8-*-

import os
import torch
import pickle

from inversion import utils
from inversion import functions


def get_transformer_block_stats(model, samples, img_size, out_path, on_cuda):
    train_stat = model.training
    model.eval()
    # register hooks:
    stat_hooks = []
    for n, mi in model.named_modules():
        if 'transformer.resblocks.' in n and len(n.split('.')) == 3:
            stat_hooks.append(InputStatHook(module_name=n, module=mi, batch_dim=1))
    normalizer = functions.Normalization(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    if on_cuda:
        normalizer.cuda()
    steps = int(samples // 100)
    if samples % 100 != 0:
        steps += 1
    with torch.no_grad():
        for i in range(steps):
            start = i * 100
            end = min((i + 1) * 100, samples)
            rand_input = torch.rand([end - start] + img_size, dtype=torch.float32, requires_grad=False)
            if on_cuda:
                rand_input = rand_input.cuda()
            rand_input = normalizer(rand_input)
            _ = model(rand_input)
            for hi in stat_hooks:
                hi.update_stat()
    model.train(train_stat)
    name2stat = {}
    for hi in stat_hooks:
        name = hi.get_module_name()
        mean, var = hi.compute_stat()
        name2stat[name] = [mean, var]
        hi.remove_hook()
    out_file = os.path.join(out_path, 'model_input_stats.pkl')
    with open(out_file, 'wb') as fw:
        pickle.dump(name2stat, fw)


def get_data_transformer_block_stats(model, data_loader, out_path, on_cuda, fname='model_input_stats.pkl'):
    train_stat = model.training
    model.eval()
    # register hooks:
    stat_hooks = []
    for n, mi in model.named_modules():
        if 'transformer.resblocks.' in n and len(n.split('.')) == 3:
            stat_hooks.append(InputStatHook(module_name=n, module=mi, batch_dim=1))
    total_cnt = 0
    with torch.no_grad():
        for data in data_loader:
            idx, sp, lab = data
            if on_cuda:
                sp = sp.cuda()
            _ = model(sp)
            for hi in stat_hooks:
                hi.update_stat()
            total_cnt += sp.shape[0]
    model.train(train_stat)
    name2stat = {}
    for hi in stat_hooks:
        name = hi.get_module_name()
        mean, var = hi.compute_stat()
        name2stat[name] = [mean, var, total_cnt]
        hi.remove_hook()
    out_file = os.path.join(out_path, fname)
    with open(out_file, 'wb') as fw:
        pickle.dump(name2stat, fw)


def get_block_input_stats(model, samples, img_size, out_path, on_cuda, split_cnn=False):
    all_blocks, batch_dims = utils.split_clip_blocks(model=model, normalize=True, split_cnn=split_cnn)
    # add block hooks
    bid2hookds = {}
    for i, bi in enumerate(all_blocks):
        if i == 0:
            continue
        bid2hookds[i] = [InputStatHook(module_name='total', module=bi, batch_dim=batch_dims[i])]
        for n, mi in bi.named_modules():
            if n == 'mlp' or n == 'attn':
                bid2hookds[i].append(InputStatHook(module_name=n, module=mi, batch_dim=batch_dims[i]))
    temp_model = torch.nn.Sequential(*all_blocks)
    normalizer = functions.Normalization(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    if on_cuda:
        normalizer.cuda()
    steps = int(samples // 100)
    if samples % 100 != 0:
        steps += 1
    with torch.no_grad():
        for i in range(steps):
            start = i * 100
            end = min((i + 1) * 100, samples)
            rand_input = torch.rand([end - start] + img_size, dtype=torch.float32, requires_grad=False)
            if on_cuda:
                rand_input = rand_input.cuda()
            rand_input = normalizer(rand_input)
            _ = temp_model(rand_input)
            for bid in bid2hookds.keys():
                for hi in bid2hookds[bid]:
                    hi.update_stat()
    for bid in bid2hookds.keys():
        out_file = os.path.join(out_path, 'block_input_stats' + str(bid) + '.pkl')
        name2stat = {}
        for hi in bid2hookds[bid]:
            name = hi.get_module_name()
            mean, var = hi.compute_stat()
            name2stat[name] = [mean, var]
            hi.remove_hook()
        with open(out_file, 'wb') as fw:
            pickle.dump(name2stat, fw)


def get_data_block_input_stats(model, data_loader, out_path, on_cuda, fname='block_input_stats', split_cnn=False):
    all_blocks, batch_dims = utils.split_clip_blocks(model=model, normalize=True, split_cnn=split_cnn)
    # add block hooks
    bid2hookds = {}
    for i, bi in enumerate(all_blocks):
        if i == 0:
            continue
        bid2hookds[i] = [InputStatHook(module_name='total', module=bi, batch_dim=batch_dims[i])]
        for n, mi in bi.named_modules():
            if n == 'mlp' or n == 'attn':
                bid2hookds[i].append(InputStatHook(module_name=n, module=mi, batch_dim=batch_dims[i]))
    temp_model = torch.nn.Sequential(*all_blocks)
    total_cnt = 0
    with torch.no_grad():
        for data in data_loader:
            idx, sp, lab = data
            if on_cuda:
                sp = sp.cuda()
            _ = temp_model(sp)
            for bid in bid2hookds.keys():
                for hi in bid2hookds[bid]:
                    hi.update_stat()
            total_cnt += sp.shape[0]
    for bid in bid2hookds.keys():
        out_file = os.path.join(out_path, fname + str(bid) + '.pkl')
        name2stat = {}
        for hi in bid2hookds[bid]:
            name = hi.get_module_name()
            mean, var = hi.compute_stat()
            name2stat[name] = [mean, var, total_cnt]
            hi.remove_hook()
        with open(out_file, 'wb') as fw:
            pickle.dump(name2stat, fw)


def merge_stats(ori_stat_file, new_stat_file):
    with open(ori_stat_file, 'rb') as fr:
        ori_name2stat = pickle.load(fr)
    with open(new_stat_file, 'rb') as fr:
        new_name2stat = pickle.load(fr)
    name2stat = {}
    for name in ori_name2stat.keys():
        ori_mean, ori_var, ori_cnt = ori_name2stat[name]
        new_mean, new_var, new_cnt = new_name2stat[name]
        mean = ori_cnt / (ori_cnt + new_cnt) * ori_mean + new_cnt / (ori_cnt + new_cnt) * new_mean
        ori_mean_square = ori_var + ori_mean ** 2
        new_mean_square = new_var + new_mean ** 2
        mean_square = ori_cnt / (ori_cnt + new_cnt) * ori_mean_square \
            + new_cnt / (ori_cnt + new_cnt) * new_mean_square
        var = mean_square - mean ** 2
        name2stat[name] = [mean, var, ori_cnt + new_cnt]
    with open(ori_stat_file, 'wb') as fw:
        pickle.dump(name2stat, fw)


class InputStatHook(object):
    def __init__(self, module, module_name, batch_dim):
        self.module = module
        self.module_name = module_name
        self.inputs = None
        self.handle = self.module.register_forward_hook(hook=self.get_input_hook())
        self.cnt = 0
        self.mean = 0
        self.mean_square = 0
        self.var = 0
        self.batch_dim = batch_dim

    def update_stat(self):
        # get stats of input
        batch_size = self.inputs.shape[0]
        if len(self.inputs.shape) == 4:  # consider 4-dim input
            mean = self.inputs.mean([self.batch_dim, 2, 3]).detach().cpu()
            mean_square = (self.inputs ** 2).mean([self.batch_dim, 2, 3]).detach().cpu()
        else:
            mean = self.inputs.mean([self.batch_dim, 2]).detach().cpu()
            mean_square = (self.inputs ** 2).mean([self.batch_dim, 2]).detach().cpu()
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
