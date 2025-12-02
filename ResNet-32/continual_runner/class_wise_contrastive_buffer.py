# -*-coding:utf8-*-

import os
import pickle
import random
import torch
from torch.utils.data import DataLoader, Dataset
import copy
import torchvision
import numpy as np
import time
import torch.nn.functional as F

from inversion import class_wise_image_for_contrastive
from datasets import single_task_dataset
from backbone import vaes


class ClassWiseContrastiveBuffer(object):
    def __init__(self, local_path, teacher_model, dataset, inversion_params, buffer_params,
                 cont_params, eval_epoch=40, relabel_input=False):
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.dataset = dataset
        self.teacher_model = teacher_model
        self.inversion_params = inversion_params
        self.buffer_params = buffer_params
        self.init_samples = self.buffer_params['init_samples']
        self.per_task_samples = self.buffer_params['per_task_samples']
        self.gen_batch_size = self.buffer_params['gen_batch_size']
        self.cont_params = cont_params
        self.eval_epoch = eval_epoch
        self.relabel_input = relabel_input
        if dataset == 'seq_cifar100':
            image_size = [3, 32, 32]
        elif dataset == 'seq_tinyimagenet':
            image_size = [3, 64, 64]
        else:
            raise ValueError('Invalid dataset type')
        self.cls2mean = {}
        self.cls2std = {}
        self.logit_cls2mean = {}
        self.logit_cls2std = {}
        self.generator = class_wise_image_for_contrastive.ClassWiseContrastiveInversion(
            image_size=image_size,
            model=copy.deepcopy(self.teacher_model),
            lr=self.inversion_params['lr'],
            train_steps=self.inversion_params['train_steps'],
            alpha_pr=self.inversion_params['alpha_pr'],
            alpha_rf=self.inversion_params['alpha_rf'],
            local_path=os.path.join(self.local_path, 'generator'),
            flip_rate=0 if 'flip_rate' not in self.inversion_params else self.inversion_params['flip_rate'],
            scheduler_params=self.inversion_params['scheduler_params'],
            log_step=1000 if 'log_step' not in self.inversion_params else self.inversion_params['log_step'],
            hook_type='kl',
            hook_rate=1.0,
            mse_type='l2',
            opt_type='adam',
            boost_factor=self.inversion_params['boost_factor'],
        )
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.kd_loss = torch.nn.MSELoss()
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()
        self.train_samples = None
        self.train_labels = None
        self.train_logits = []
        self.unslt_samples = None
        self.unslt_labels = None
        self.seen_tasks = 0
        self.cont_models = []

    def generate_features(self, samples):
        labs = np.random.randint(low=0, high=self.teacher_model.head.num_classes, size=[samples])
        lab2cnt = {}
        for li in labs:
            if int(li) not in lab2cnt:
                lab2cnt[int(li)] = 1
            else:
                lab2cnt[int(li)] += 1
        all_feats = []
        all_labs = []
        for li in lab2cnt.keys():
            cls_mean = self.cls2mean[li]
            cls_std = self.cls2std[li]
            if self.buffer_params['feat_type'] == 'prv_feat':
                cls_feats = from_previous_feature(out_path=self.local_path, lab=li, samples=lab2cnt[li])
            elif self.buffer_params['feat_type'] == 'cont':
                batch_size = max(int(lab2cnt[li] // self.cont_params['slt_step']), 2)
                cls_feats = contrastive_selection(
                    cont_model=self.cont_models[li],
                    cls_mean=cls_mean,
                    cls_std=cls_std,
                    samples=lab2cnt[li],
                    batch_size=batch_size,
                    slt_rate=min(self.cont_params['slt_rate'], 1.0),
                    feat_dim=cls_mean.shape[1],
                    on_cuda=self.buffer_params['use_cuda'],
                    tau=self.cont_params['tau'],
                    cur_feats=None
                )
            else:
                cls_feats = random_selection(
                    cls_mean=cls_mean,
                    cls_std=cls_std,
                    samples=lab2cnt[li],
                    feat_dim=cls_mean.shape[1]
                )
            all_feats.append(cls_feats)
            labs = np.ones([lab2cnt[li]]) * li
            all_labs.append(labs)
        all_feats = torch.cat(all_feats, dim=0)
        all_labs = torch.tensor(np.concatenate(all_labs, axis=0), dtype=torch.long, requires_grad=False)
        # shuffle features and targets
        rand_ids = list(range(samples))
        random.shuffle(rand_ids)
        rand_ids = np.array(rand_ids)
        all_feats = all_feats[rand_ids, :]
        all_labs = all_labs[rand_ids]
        return all_feats, all_labs

    def generate_data_by_selection(self, selection_agent, samples):
        # generate data based on new class selection.
        print('start generation time:', time.ctime())
        if self.buffer_params['use_cuda']:
            self.generator.cuda()
        train_samples = []
        train_labels = []
        train_cnt = 0
        used_feats = 0
        step = 0
        # sample all features at once
        num_feats = samples
        if selection_agent is not None:  # over sample in features.
            interfere_rate = selection_agent.slt_rate
            num_feats = int(samples * int(1.0 / interfere_rate + 1))
            num_feats = max(num_feats, int(samples / interfere_rate) + 200)
        else:
            interfere_rate = 1.0
        all_target_feats, all_targets = self.generate_features(samples=num_feats)
        all_targets = all_targets.cuda()
        while train_cnt < samples:
            iters = self.inversion_params['train_steps']
            bs = min(self.inversion_params['layer_batch'],
                     max(int((samples - train_cnt) / interfere_rate), 200))
            if bs % self.gen_batch_size != 0 and bs % self.gen_batch_size < 200:
                # small batch may lead to poor performance on distribution loss
                bs += 200
            inputs, _ = self.generator.layer_wise_inversion_for_cl(
                batch_size=min(self.gen_batch_size, bs),
                finetune_iters=self.inversion_params['tune_steps'],
                finetune_lr=self.inversion_params['tune_lr'],
                target_feats=all_target_feats[used_feats:used_feats + bs, :],
                target=all_targets[used_feats:used_feats + bs],
                return_best=True,
                iters=iters,
                rf_factor=self.inversion_params['rf_factor'],
                search_param=self.inversion_params['search_param'],
                selection_agent=selection_agent
            )
            inputs = inputs.detach()
            targets = copy.deepcopy(all_targets[used_feats:used_feats + bs])
            used_feats += inputs.shape[0]
            slt_inputs = inputs
            slt_targets = targets
            if slt_inputs.shape[0] + train_cnt > samples:
                train_samples.append(slt_inputs[:samples - train_cnt, :])
                train_labels.append(slt_targets[:samples - train_cnt])
            else:
                train_samples.append(slt_inputs)
                train_labels.append(slt_targets)
            train_cnt += train_samples[-1].shape[0]
            print('generate samples:', train_cnt)
            step += 1
        if self.train_samples is not None and self.train_labels is not None:
            train_samples = torch.cat(train_samples, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            self.train_samples = torch.cat([self.train_samples, train_samples], dim=0)
            self.train_labels = torch.cat([self.train_labels, train_labels], dim=0)
        else:
            self.train_samples = torch.cat(train_samples, dim=0)
            self.train_labels = torch.cat(train_labels, dim=0)
        print('end generation time:', time.ctime())
        self.compute_teacher_logit()

    def get_data(self, batch_size):
        inds = random.sample(list(range(self.train_samples.shape[0])), batch_size)
        batch_sp = []
        batch_lab = []
        teacher_logit = []
        for ind in inds:
            batch_sp.append(self.train_samples[ind, :])
            batch_lab.append(self.train_labels[ind])
            teacher_logit.append(self.train_logits[ind, :])
        batch_sp = torch.stack(batch_sp, dim=0)
        batch_lab = torch.stack(batch_lab, dim=0)
        teacher_logit = torch.stack(teacher_logit, dim=0)
        return batch_sp, batch_lab, teacher_logit

    def update_buffer(self, teacher_model, train_loader):
        self.seen_tasks += 1
        old_class = len(self.cont_models)
        new_class = teacher_model.head.num_classes
        cls2mean = {}
        cls2std = {}
        logit_cls2mean = {}
        logit_cls2std = {}
        # train contrastive model on old classes
        if self.train_samples is not None and self.train_labels is not None:
            buffer_loader = self.make_data_loader()
            prv_target_class = set(list(range(old_class)))
            prv_out_files, prv_lab2stat, prv_logit_stat, ori_feat_files = prepare_out_features(
                teacher_model=teacher_model,
                data_loader=buffer_loader,
                out_path=self.local_path,
                on_cuda=self.buffer_params['use_cuda'],
                target_class=prv_target_class,
                ori_teacher_model=self.teacher_model
            )
            for li in prv_lab2stat.keys():
                cls2mean[li] = torch.unsqueeze(
                    torch.tensor(prv_lab2stat[li][0], dtype=torch.float32, requires_grad=False), dim=0)
                cls2std[li] = torch.unsqueeze(
                    torch.tensor(prv_lab2stat[li][1], dtype=torch.float32, requires_grad=False), dim=0)
                logit_cls2mean[li] = prv_logit_stat[li][0]
                logit_cls2std[li] = prv_logit_stat[li][1]
            for i in range(old_class):
                ori_feat_file = os.path.join(self.local_path, 'ori_feat_' + str(i) + '.pkl')
                if os.path.exists(ori_feat_file):
                    ori_feat = []
                    with open(ori_feat_file, 'rb') as fr:
                        while True:
                            try:
                                feat, lab = pickle.load(fr)
                                ori_feat.append(feat)
                            except EOFError:
                                break
                    ori_feat = torch.stack(ori_feat, dim=0)
                    del ori_feat
                    os.remove(ori_feat_file)
                print('train contrastive model for class:', i)
                cont_model = train_contrastive_model(
                    data_file=prv_out_files[i],
                    use_cuda=self.buffer_params['use_cuda'],
                    model_params=self.cont_params,
                    act='leaky' if 'act' not in self.cont_params else self.cont_params['act'],
                    verbose=False
                )
                self.cont_models[i] = cont_model
                if ('prv_feat' not in self.buffer_params) or (not self.buffer_params['prv_feat']):
                    os.remove(prv_out_files[i])
        # train contrastive model on new classes
        target_class = set(list(range(old_class, new_class)))
        cur_out_files, cur_lab2stat, cur_logit_stat, _ = prepare_out_features(
            teacher_model=teacher_model,
            data_loader=train_loader,
            out_path=self.local_path,
            on_cuda=self.buffer_params['use_cuda'],
            target_class=target_class,
            ori_teacher_model=None
        )
        for li in cur_lab2stat.keys():
            cls2mean[li] = torch.unsqueeze(
                torch.tensor(cur_lab2stat[li][0], dtype=torch.float32, requires_grad=False), dim=0)
            cls2std[li] = torch.unsqueeze(
                torch.tensor(cur_lab2stat[li][1], dtype=torch.float32, requires_grad=False), dim=0)
            logit_cls2mean[li] = cur_logit_stat[li][0]
            logit_cls2std[li] = cur_logit_stat[li][1]
        for i in range(old_class, new_class):
            print('train contrastive model for class:', i)
            cont_model = train_contrastive_model(
                data_file=cur_out_files[i],
                use_cuda=self.buffer_params['use_cuda'],
                model_params=self.cont_params,
                act='leaky' if 'act' not in self.cont_params else self.cont_params['act'],
                verbose=False
            )
            self.cont_models.append(cont_model)
            if ('prv_feat' not in self.buffer_params) or (not self.buffer_params['prv_feat']):
                os.remove(cur_out_files[i])
        self.cls2mean = cls2mean
        self.cls2std = cls2std
        self.logit_cls2mean = logit_cls2mean
        self.logit_cls2std = logit_cls2std
        # update generator and samples
        self.teacher_model = None
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.train_samples = None
        self.train_labels = None
        self.unslt_samples = None
        self.unslt_labels = None
        self.generator.update_model(model=copy.deepcopy(self.teacher_model))

    def make_data_loader(self):  # for computing class mean and std for buffer data
        data = []
        for i in range(self.train_samples.shape[0]):
            data.append([self.train_samples[i, :], self.train_labels[i]])
        if self.unslt_samples is not None and self.unslt_labels is not None:
            for i in range(self.unslt_samples.shape[0]):
                data.append([self.unslt_samples[i, :], self.unslt_labels[i]])
        dataset = single_task_dataset.SimpleDataset(data=data, transforms=None)
        data_loader = DataLoader(dataset, batch_size=100, drop_last=False)
        return data_loader

    def get_teacher_model(self):
        return self.teacher_model

    def dump_data(self, task_id):
        save_path = os.path.join(self.local_path, 'inv_train_data' + str(task_id) + '.pkl')
        with open(save_path, 'wb') as fw:
            pickle.dump(self.train_samples, fw)
            pickle.dump(self.train_labels, fw)

    def prepare_for_save(self):
        self.train_samples = None
        self.train_labels = None
        self.unslt_samples = None
        self.unslt_labels = None

    def compute_teacher_logit(self):
        self.train_logits = []
        batch_size = 100
        steps = self.train_samples.shape[0] // batch_size
        if self.train_samples.shape[0] % batch_size != 0:
            steps += 1
        for i in range(steps):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.train_samples.shape[0])
            batch = self.train_samples[start:end, :]
            if bool(self.buffer_params['use_cuda']):
                batch = batch.cuda()
            batch_logits = self.teacher_model(batch).clone().detach()
            self.train_logits.append(batch_logits)
        self.train_logits = torch.cat(self.train_logits, dim=0)


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
    loss_fn = vaes.NegativeContrastiveLoss(tau=model_params['tau'])
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


def prepare_out_features(teacher_model, data_loader, out_path, on_cuda, target_class, ori_teacher_model=None):
    teacher_model.eval()
    out_files = {}
    ori_out_files = {}
    lab2feat = {}
    ori_lab2feat = {}
    lab2stat = {}
    lab2logit = {}
    logit_lab2stat = {}
    for li in target_class:
        lab2feat[li] = []
        lab2logit[li] = []
        ori_lab2feat[li] = []
    pool_layer = torch.nn.AdaptiveAvgPool2d(1)
    with torch.no_grad():
        for data in data_loader:
            sp, lab = data
            if on_cuda:
                sp = sp.cuda()
            lab = lab.cpu().numpy()
            out_feat = teacher_model.backbone(sp).detach()
            if ori_teacher_model is not None:
                ori_out_feat = pool_layer(ori_teacher_model.backbone(sp))[:, :, 0, 0].detach().cpu()
            out_logit = teacher_model.head(out_feat).detach()
            out_feat = pool_layer(out_feat)[:, :, 0, 0]
            out_feat = out_feat.cpu().numpy()
            out_logit = out_logit.cpu().numpy()
            for i in range(sp.shape[0]):
                lab2feat[int(lab[i])].append(out_feat[i, :])
                lab2logit[int(lab[i])].append(out_logit[i, :])
                if ori_teacher_model is not None:
                    ori_lab2feat[int(lab[i])].append(ori_out_feat[i, :])
    for lab in lab2feat.keys():
        out_file = os.path.join(out_path, 'feat_' + str(lab) + '.pkl')
        if os.path.exists(out_file):
            os.remove(out_file)
        with open(out_file, 'wb') as fw:
            for i in range(len(lab2feat[lab])):
                pickle.dump([lab2feat[lab][i], lab], fw)
        out_files[lab] = out_file
        if ori_teacher_model is not None:
            ori_out_file = os.path.join(out_path, 'ori_feat_' + str(lab) + '.pkl')
            if os.path.exists(ori_out_file):
                os.remove(ori_out_file)
            with open(ori_out_file, 'wb') as fw:
                for i in range(len(ori_lab2feat[lab])):
                    pickle.dump([ori_lab2feat[lab][i], lab], fw)
            ori_out_files[lab] = ori_out_file
        lab_feats = np.stack(lab2feat[lab], axis=0)
        mean = np.mean(lab_feats, axis=0)
        std = np.std(lab_feats, axis=0)
        lab2stat[lab] = [mean, std]
        lab_logits = np.stack(lab2logit[lab], axis=0)
        logit_mean = np.mean(lab_logits, axis=0)
        logit_std = np.std(lab_logits, axis=0)
        logit_lab2stat[lab] = [logit_mean, logit_std]
    return out_files, lab2stat, logit_lab2stat, ori_out_files


def contrastive_selection(cont_model, cls_mean, cls_std, samples, batch_size, slt_rate, feat_dim, on_cuda,
                          tau=1.0, cur_feats=None):
    # mean and std are torch.Tensor
    all_feats = cur_feats
    cur_num = 0
    if cur_feats is not None:  # add function for reference features
        cur_num = cur_feats.shape[0]
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
    # build loss function
    for i in range(steps):
        if all_feats is None:
            eps = torch.randn([min(batch_size, samples), feat_dim], dtype=torch.float32, requires_grad=False)
        else:
            eps = torch.randn([sup_batch, feat_dim], dtype=torch.float32, requires_grad=False)
        feats = eps * cls_std + cls_mean
        if all_feats is None:
            all_feats = feats
            continue
        # rand_ids = random.sample(list(range(all_feats.shape[0])), min(int(samples * 0.5), all_feats.shape[0]))
        # other_feats = all_feats[np.array(rand_ids), :]
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
    all_feats = all_feats[cur_num:, :]
    # add function for compensating mean and std
    # aims to ensure selected feature have same statistics as real feature
    feats_mean = torch.mean(all_feats, dim=0)
    feats_std = torch.std(all_feats, dim=0)
    bias = cls_mean - feats_mean
    scale = cls_std / (feats_std + 1e-8)
    all_feats = (all_feats + bias) * scale
    return all_feats


def random_selection(cls_mean, cls_std, samples, feat_dim):
    eps = torch.randn([samples, feat_dim], dtype=torch.float32, requires_grad=False)
    feats = eps * cls_std + cls_mean
    return feats


def from_previous_feature(out_path, lab, samples):
    out_file = os.path.join(out_path, 'feat_' + str(lab) + '.pkl')
    feats = []
    with open(out_file, 'rb') as fr:
        while True:
            try:
                feat, li = pickle.load(fr)
                feats.append(feat)
            except EOFError:
                break
    feats = np.stack(feats, axis=0)
    rand_feats = np.random.randint(low=0, high=feats.shape[0], size=[samples])
    all_feats = feats[rand_feats, :]
    all_feats = torch.tensor(all_feats, dtype=torch.float32, requires_grad=False)
    return all_feats
