# -*-coding:utf8-*-

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import copy
import numpy as np


class SimpleMLPEncoder(torch.nn.Module):
    def __init__(self, in_features, blocks, act_type='leaky'):
        super().__init__()
        self.n_ch = in_features
        self.blocks = []
        for i in range(blocks):
            block = torch.nn.Linear(in_features=self.n_ch, out_features=self.n_ch, bias=True)
            self.blocks.append(block)
            bn = torch.nn.BatchNorm1d(num_features=self.n_ch)
            self.blocks.append(bn)
            if act_type == 'leaky':
                act = torch.nn.LeakyReLU()
            else:
                act = torch.nn.ReLU()
            self.blocks.append(act)
        self.encoder = torch.nn.Sequential(*self.blocks)

    def forward(self, x):
        y = self.encoder(x)
        y = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-6)
        return y


class FeatureDataset(Dataset):
    def __init__(self, data_file):
        self.all_data = []
        with open(data_file, 'rb') as fr:
            while True:
                try:
                    feat = pickle.load(fr)
                    self.all_data.append(feat)
                except EOFError:
                    break

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        return torch.tensor(self.all_data[index][0], dtype=torch.float32, requires_grad=False), self.all_data[index][1]

    def get_shape(self):
        return self.all_data[0][0].shape


class SelectContrastiveLoss(torch.nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x, y):  # inputs are normalized
        x_1 = torch.unsqueeze(x, dim=1)  # shape = [N, 1, dim]
        x_2 = torch.unsqueeze(y, dim=0)  # shape = [1, M, dim]
        cos = torch.sum(x_1 * x_2, dim=2) / self.tau  # shape = [N, M]
        loss = torch.mean(cos, dim=1)
        return loss


class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x):  # x.shape = [N, dim]
        x_1 = torch.unsqueeze(x, dim=0)  # shape = [1, N, dim]
        x_2 = torch.unsqueeze(x, dim=1)  # shape = [N, 1, dim]
        cos = torch.sum(x_1 * x_2, dim=2) / self.tau  # shape = [N, N]
        exp_cos = torch.exp(cos)
        loss = torch.log(torch.mean(exp_cos, dim=1))  # diagonal elements are positive pairs
        loss = torch.mean(loss)
        return loss


def train_contrastive_model(data_file, use_cuda, model_params, act='leaky', verbose=True):
    train_dataset = FeatureDataset(data_file=data_file)
    if len(train_dataset) <= 1:
        print('\tno features for training contrastive model')
    in_shape = train_dataset.get_shape()
    train_loader = DataLoader(train_dataset, batch_size=min(64, len(train_dataset)), drop_last=True, shuffle=True)
    model = SimpleMLPEncoder(blocks=model_params['blocks'], in_features=in_shape[0], act_type=act)
    model.train()
    if use_cuda:
        model = model.cuda()
    opt = torch.optim.Adam(lr=model_params['lr'], params=model.parameters())
    loss_fn = NegativeContrastiveLoss(tau=model_params['tau'])
    best_loss = None
    best_model = None
    for e in range(model_params['epoch']):
        step = 0
        for data in train_loader:
            sp, lab = data
            if use_cuda:
                sp = sp.cuda()
            out = model(sp)
            cont_loss = loss_fn(out)
            if bool(torch.isnan(cont_loss)):
                print('\tthe loss is NAN during training VAE', 'epoch:', e, 'step:', step)
                break
            opt.zero_grad()
            cont_loss.backward()
            opt.step()
            if best_loss is None or cont_loss.item() < best_loss:
                best_loss = cont_loss.item()
                best_model = copy.deepcopy(model)
            if step % 100 == 0 and verbose:
                print('step', step, '\t', 'contrastive_loss:', cont_loss.item())
            step += 1
        if e % 10 == 0 and verbose:
            print('finish training epoch', e)
    print('\tbest loss is:', best_loss)
    del train_loader
    del train_dataset
    if best_model is None:
        best_model = model
    best_model.eval()
    return best_model


def contrastive_selection(cont_model, stats, samples, batch_size, slt_rate, feat_dim, on_cuda, tau=1.0):
    # mean and std are torch.Tensor
    all_feats = None
    # iterative selection
    steps = int(samples // batch_size)
    if samples % batch_size != 0:
        steps += 1
    cont_loss = SelectContrastiveLoss(tau=tau)
    if slt_rate > 0:
        sup_batch = int(batch_size / slt_rate)
    else:
        sup_batch = batch_size
    mean_cont_loss = 0
    slt_cont_loss = 0
    all_cand_feats = []
    # build loss function
    for i in range(steps):
        if isinstance(stats, list) or isinstance(stats, tuple):
            cls_mean, cls_std = stats
            if all_feats is None:
                eps = torch.randn([min(batch_size, samples), feat_dim], dtype=torch.float32, requires_grad=False)
            else:
                eps = torch.randn([sup_batch, feat_dim], dtype=torch.float32, requires_grad=False)
            feats = eps * cls_std + cls_mean
        else:
            feats, _ = stats.sample(n_samples=min(batch_size, samples) if all_feats is None else sup_batch)
            all_cand_feats.append(copy.deepcopy(feats))
            feats = torch.tensor(feats, dtype=torch.float32, requires_grad=False)
        if all_feats is None:
            all_feats = feats
            continue
        other_feats = all_feats
        if on_cuda:
            feats = feats.cuda()
            other_feats = other_feats.cuda()
        with torch.no_grad():
            cont_out = cont_model(feats)
            other_out = cont_model(other_feats)
            losses = cont_loss(cont_out, other_out).detach().cpu().numpy()
        mean_cont_loss += np.sum(losses)
        slt_ids = np.argsort(losses)[:min(batch_size, samples - all_feats.shape[0])]
        slt_cont_loss += np.sum(losses[slt_ids])
        if on_cuda:
            feats = feats.cpu()
            all_feats = all_feats.cpu()
        slt_feats = feats[slt_ids, :]
        all_feats = torch.cat([all_feats, slt_feats], dim=0)
        if all_feats.shape[0] == samples:
            break
    # add function for compensating mean and std
    # aims to ensure selected feature have same statistics as real feature
    feats_mean = torch.mean(all_feats, dim=0)
    feats_std = torch.std(all_feats, dim=0)
    if isinstance(stats, list) or isinstance(stats, tuple):
        cls_mean, cls_std = stats
    else:
        all_cand_feats = torch.tensor(np.concatenate(all_cand_feats, axis=0), dtype=torch.float32, requires_grad=False)
        cls_mean = torch.mean(all_cand_feats, dim=0)
        cls_std = torch.std(all_cand_feats, dim=0)
    bias = cls_mean - feats_mean
    scale = cls_std / (feats_std + 1e-8)
    all_feats = (all_feats + bias) * scale
    return all_feats
