# -*-coding:utf8-*-

import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from inversion import utils


def get_class_wise_distribution(backbone, data_loader, classes, on_cuda):
    train_stat = backbone.training
    backbone.eval()
    cls2feats = {}
    for ci in classes:
        cls2feats[ci] = []
    for data in data_loader:
        idx, sp, lab = data
        if on_cuda:
            sp = sp.cuda()
        out_feat = backbone(sp)[0]
        out_feat = out_feat.detach()
        if on_cuda:
            out_feat = out_feat.cpu()
        out_feat = out_feat.numpy()
        for i in range(sp.shape[0]):
            cls = int(lab[i])
            cls2feats[cls].append(out_feat[i, :])
    cls2stat = {}
    for ci in cls2feats.keys():
        all_feats = np.stack(cls2feats[ci], axis=0)
        cls_mean = torch.unsqueeze(
            torch.tensor(np.mean(all_feats, axis=0), dtype=torch.float32, requires_grad=False), dim=0)
        cls_std = torch.unsqueeze(
            torch.tensor(np.std(all_feats, axis=0), dtype=torch.float32, requires_grad=False), dim=0)
        cls2stat[ci] = [cls_mean, cls_std]
    backbone.train(train_stat)
    return cls2stat, cls2feats


def get_gmm_class_distribution(backbone, data_loader, classes, on_cuda, components=3, reg_covar=1e-6):
    # GMM modelling for class feature distribution
    train_stat = backbone.training
    backbone.eval()
    cls2feats = {}
    for ci in classes:
        cls2feats[ci] = []
    for data in data_loader:
        idx, sp, lab = data
        if on_cuda:
            sp = sp.cuda()
        out_feat = backbone(sp)[0]
        out_feat = out_feat.detach()
        if on_cuda:
            out_feat = out_feat.cpu()
        out_feat = out_feat.numpy()
        for i in range(sp.shape[0]):
            cls = int(lab[i])
            cls2feats[cls].append(out_feat[i, :])
    cls2stat = {}
    for ci in cls2feats.keys():
        all_feats = np.stack(cls2feats[ci], axis=0)
        cls_gmm = GaussianMixture(n_components=components, covariance_type='diag', reg_covar=reg_covar)
        cls_gmm.fit(all_feats)
        cls2stat[ci] = cls_gmm
    backbone.train(train_stat)
    return cls2stat, cls2feats


def update_old_centers(cls2stat, backbone, start_block, normalize, samples, labels, on_cuda, momentum):
    steps = int(len(samples) // 100)
    if len(samples) % 100 != 0:
        steps += 1
    cls2feats = {}
    for ci in cls2stat.keys():
        cls2feats[ci] = []
    if start_block > 0:
        temp_blocks, batch_dim = utils.split_clip_blocks(model=backbone, split_cnn=True, normalize=normalize)
        temp_model = torch.nn.Sequential(*temp_blocks[start_block:])
    else:
        temp_model = backbone
    for i in range(steps):
        start = i * 100
        end = min((i + 1) * 100, len(samples))
        sps = torch.stack(samples[start:end], dim=0)
        labs = labels[start:end]
        if on_cuda:
            sps = sps.cuda()
        out_feat = temp_model(sps).detach().cpu().numpy()
        for j in range(sps.shape[0]):
            lab = int(labs[j])
            cls2feats[lab] = out_feat[j, :]
    new_cls2stat = []
    for ci in cls2feats.keys():
        all_feats = np.stack(cls2feats[ci], axis=0)
        cur_mean = torch.unsqueeze(
            torch.tensor(np.mean(all_feats, axis=0), dtype=torch.float32, requires_grad=False), dim=0)
        old_mean, old_std = cls2stat[ci]
        new_mean = momentum * old_mean + (1 - momentum) * cur_mean
        new_cls2stat[ci] = [new_mean, old_std]
    return new_cls2stat


class PromptVitWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x, pen=True, train=False)
        return out, None


class PromptQueryWrapper(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.model = vit
        self.input_patchnorm = self.model.input_patchnorm
        self.grid_size = vit.grid_size
        self.patch_size = vit.patch_size
        self.patchnorm_pre_ln = vit.patchnorm_pre_ln
        self.conv1 = vit.conv1
        self.class_embedding = vit.class_embedding
        self.positional_embedding = vit.positional_embedding
        self.patch_dropout = vit.patch_dropout
        self.ln_pre = vit.ln_pre
        # self.transformer = vit.transformer
        self.transformer_blocks = vit.transformer.resblocks
        self.attn_pool = vit.attn_pool
        self.ln_post = vit.ln_post
        self._global_pool = vit._global_pool
        self.output_tokens = vit.output_tokens
        self.proj = vit.proj
        self.embed_dim = 768
        self.feature_dim = 768

    def forward(self, x):
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0],
                self.grid_size[1], self.patch_size[1]
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i, blk in enumerate(self.transformer_blocks):
            x = blk(x, layer=i, prompt=None)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 0, :]
        return x, None
