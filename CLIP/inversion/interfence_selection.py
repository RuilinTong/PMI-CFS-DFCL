# -*-coding:utf8-*-

import torch
import numpy as np


class InterferenceSelectionAgent(object):
    def __init__(self, tuned_blocks, ori_blocks, layer_names, select_rate, device, text_feat, ori_text_feat, know_class,
                 logit_scale, batch_dim, head=None, slt_mode='loss_inc'):
        self.tuned_blocks = tuned_blocks  # with normalization
        self.ori_blocks = ori_blocks  # with normalization
        self.layer_names = layer_names
        self.num_blocks = len(tuned_blocks)
        self.select_rate = select_rate
        self.device = device
        self.text_feat = text_feat
        self.ori_text_feat = ori_text_feat
        self.know_class = know_class
        self.logit_scale = logit_scale
        self.batch_dim = batch_dim
        self.head = head
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.slt_mode = slt_mode

    def select_by_interference(self, in_feats, targets):
        self.text_feat = self.text_feat.to(self.device)
        self.ori_text_feat = self.ori_text_feat.to(self.device)
        in_feats, targets = in_feats.to(self.device), targets.to(self.device)
        if len(self.tuned_blocks) > 0 and len(self.ori_blocks) > 0:  # add support for only tune text encoder
            tuned_model = torch.nn.Sequential(*self.tuned_blocks)
            ori_model = torch.nn.Sequential(*self.ori_blocks)
        else:
            tuned_model = None
            ori_model = None
        steps = in_feats.shape[self.batch_dim] // 100
        if in_feats.shape[self.batch_dim] % 100 != 0:
            steps += 1
        loss_diffs = []
        with torch.no_grad():
            for i in range(steps):
                start = i * 100
                end = min((i + 1) * 100, in_feats.shape[self.batch_dim])
                if self.batch_dim == 0:  # consider batch dim
                    step_feats = in_feats[start:end, :]
                elif self.batch_dim == 1:
                    step_feats = in_feats[:, start:end, :]
                else:
                    raise ValueError('Batch dim should be 0 or 1')
                step_targets = targets[start:end]
                if tuned_model is not None and ori_model is not None:
                    tuned_fi = tuned_model(step_feats).detach()
                    ori_fi = ori_model(step_feats).detach()
                else:
                    tuned_fi = step_feats
                    ori_fi = step_feats
                if self.head is not None:
                    tuned_logits = self.head(tuned_fi)[:, :self.know_class]
                    ori_logits = self.head(ori_fi)[:, :self.know_class]
                else:
                    tuned_logits = self.logit_scale * tuned_fi @ self.text_feat.t()
                    ori_logits = self.logit_scale * ori_fi @ self.ori_text_feat.t()
                tuned_loss = self.loss_fn(tuned_logits, step_targets).detach()
                ori_loss = self.loss_fn(ori_logits, step_targets).detach()
                if self.slt_mode == 'loss_inc':  # add support of selection mode, loss increment or loss value
                    loss_diff = tuned_loss - ori_loss
                else:
                    loss_diff = tuned_loss
                loss_diff = loss_diff.detach().cpu().numpy()
                for j in range(loss_diff.shape[0]):
                    loss_diffs.append(loss_diff[j])
        loss_diffs = np.array(loss_diffs)
        sorted_ids = np.flip(np.argsort(loss_diffs), axis=0)
        slt_num = int(in_feats.shape[self.batch_dim] * self.select_rate)
        slt_ids = sorted_ids[:slt_num]
        return slt_ids, loss_diffs


def class_balance_selection(loss_diffs, targets, know_class, slt_num):
    cls2num = {}
    base_num = int(slt_num // know_class)
    res = slt_num - know_class * base_num
    cls2id = {}
    for i in range(know_class):
        cls2num[i] = base_num
        cls2id[i] = []
    for i in range(res):
        cls2num[i] += 1
    sorted_loss_diff = np.flip(np.argsort(loss_diffs), axis=0)
    for si in sorted_loss_diff:
        lab = targets[si]
        if len(cls2id[lab]) < cls2num[lab]:
            cls2id[lab].append(int(si))
    all_slt_ids = []
    for i in range(know_class):
        all_slt_ids = all_slt_ids + cls2id[i]
    all_slt_ids = np.array(all_slt_ids)
    return all_slt_ids
