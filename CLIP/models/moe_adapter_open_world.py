# -*-coding:utf8-*-

import logging
import pickle

import numpy
import numpy as np
import torch
import json
import time
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, cosine_lr, build_optimizer
import os
import copy
import random
import torchvision
import math
import torch.nn.functional as F
import shutil

from clip_backbones.moe_clip import create_model_and_transforms
from clip_backbones import myclip
from inversion import layer_wise_clip_inversion
from inversion import feature_stats
from inversion import cont_model
from inversion import utils


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def ema_update(model, ema_model, ema_decay):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = ema_decay * ema_param.data + (1.0 - ema_decay) * param.data


def load_des_file(des_file):
    with open(des_file, 'r', encoding='utf8') as fr:
        des = json.load(fr)
    new_des = {}
    for ci in des.keys():
        new_ci = ' '.join(ci.split('-'))
        new_ci = ' '.join(new_ci.split('_'))
        new_ci = new_ci.lower()
        new_des[new_ci] = des[ci]
    return new_des


def freeze_module(module, reverse=False):
    for param in module.parameters():
        param.requires_grad = reverse


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0}
        self._network, self.train_trfm, self.test_trfm, self.tokenizer = create_model_and_transforms(
            model_name=args['backbone_type'], pretrained=args['pretrained_weight'], design_details=design_details)
        self._network = self._network.to(self._device)
        self._network.eval()
        cnt = 0
        for k, v in self._network.named_parameters():  # freeze parameters
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False
                cnt += 1
            if "token_embedding" in k:
                v.requires_grad_(False)
            if "ZS_image_encoder" in k:
                v.requires_grad_(False)
            if "ZS_clip" in k:
                v.requires_grad_(False)
        print('frozen parameters:', cnt)
        proto_file = os.path.join(os.getcwd(), 'utils', 'proto_' + self.args['dataset'] + '.json')
        if os.path.exists(proto_file):
            self.prompt_template = load_json(proto_file)
        else:
            self.prompt_template = load_json('utils/templates.json')[args['dataset']]

        self.args = args
        self.epochs = args['tuned_epoch']
        self.batch_size = args['batch_size']
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.optimizer = None
        self.scheduler = None
        # self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8

        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))
        self.test_loader = None
        self.cur_texts = []
        self.all_texts = []
        self.full_test_loader = None
        self.buffer = None
        # build inversion runner
        self.local_path = args['local_path']
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.old_model = copy.deepcopy(self._network)
        self.old_model.float()
        self.old_model.cuda()
        freeze_module(module=self.old_model)
        self.inv_model = self.old_model.visual
        self.inversion_runner = layer_wise_clip_inversion.LayerWiseCLIPInversion(
            local_path=os.path.join(self.local_path, 'buffer'),
            model=self.inv_model,
            image_size=[int(si) for si in args['img_size'].split(',')],
            lr=args['inv_lr'],
            train_steps=args['train_steps'],
            alpha_pr=args['alpha_pr'],
            alpha_rf=args['alpha_rf'],
            scheduler_params=None,
            use_rf=True,
            smooth_type='tv',
            flip_rate=0,
            log_step=200,
            opt_type='adam',
            boost_factor=False,
            loss_type='mse',
            grad_norm=None,
            clip_input='clip',
            input_aug=bool(self.args['input_aug']),
            save_step=0,
            pre_size_change=None,
            normalize=bool(self.args['feat_norm']),
            rf_factor=self.args['rf_factor'],
            start_block=self.args['start_block']
        )
        self.inversion_runner.cuda()
        # prepare prior feature distribution
        if not bool(self.args['data_stat']):
            self.inversion_runner.get_input_stat()
        self.cls2stat = {}
        self.cont_models = []
        # hooks
        self.feature_hooks = []
        self.teacher_hooks = []
        # build raw CLIP model
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0}
        self.raw_clip = myclip.create_model(
            self.args['backbone_type'], pretrained=self.args['pretrained_weight'], design_details=design_details)
        # update inversion runner by loss function and feature normalization
        self.raw_clip.float()
        self.raw_clip.cuda()
        self.raw_clip.eval()
        freeze_module(module=self.raw_clip)
        self.openset_inversion_runner = layer_wise_clip_inversion.LayerWiseCLIPInversion(
            local_path=os.path.join(self.local_path, 'openset_inversion'),
            model=self.raw_clip.visual,
            image_size=[int(si) for si in args['img_size'].split(',')],
            lr=args['inv_lr'],
            train_steps=args['train_steps'],
            alpha_pr=args['open_pr'],
            alpha_rf=args['open_rf'],
            scheduler_params=None,
            use_rf=True,
            smooth_type='tv',
            flip_rate=0,
            log_step=200,
            opt_type='adam',
            boost_factor=False,
            loss_type=self.args['open_loss_type'],
            grad_norm=None,
            clip_input='clip',
            input_aug=bool(self.args['input_aug']),
            save_step=0,
            pre_size_change=None,
            normalize=True,
            rf_factor=self.args['rf_factor'],
            start_block=self.args['start_block']
        )
        self.openset_inversion_runner.cuda()
        self.raw_cls2stat = {}
        self.raw_cont_models = []

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager, tb_logger=None):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._memory_size > 0:
            logging.info(
                f"Rehearsal mode: training with memory size {self._memory_size}, fixed memory {self._fixed_memory}, "
                f"{self._memory_per_class} per class")
            train_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train", mode="train", appendent=None,
                trfm=self.train_trfm
            )
        else:
            logging.info("Rehearsal free mode")
            train_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="train", trfm=self.train_trfm)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test", trfm=self.test_trfm)
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        cur_class_names = data_manager.get_classnames(np.arange(self._known_classes, self._total_classes))
        self.cur_texts.clear()
        with torch.no_grad():  # change to MoE
            for li in cur_class_names:
                texts = self.args['prompt_template'].format(li)
                self.all_texts.append(texts)
                self.cur_texts.append(texts)

        if self._cur_task == 0:
            full_test_dataset = data_manager.get_dataset(
                np.arange(0, self.args['n_classes']), source="test", mode="test", trfm=self.test_trfm)
            self.full_test_loader = DataLoader(
                full_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        if 'ckpt' in self.args and self.args['ckpt'] > 0 and self.args['ckpt'] > self._cur_task:
            return

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if self._total_classes == data_manager.n_classes:
            if 'save_ckpt' in self.args and bool(self.args['save_ckpt']) and self.args['ckpt'] != self._cur_task:
                # add support for saving checkpoint
                self.save_trained_checkpoint()
            if 'ckpt' in self.args and self.args['ckpt'] > 0 and self.args['ckpt'] == self._cur_task:
                self.load_trained_checkpoint(transforms=train_dataset.trsf)
            if self.args['openset_epoch'] > 0:
                # train on open set class data
                if 'text_aug_mode' in self.args and len(self.args['text_aug_mode']) > 0 \
                        and self.args['text_aug_mode'] != 'proto_map':
                    cls2txts = self.text_augment(cur_names=cur_class_names, aug_method=self.args['text_aug_mode'])
                else:
                    des_file = os.path.join(os.getcwd(), 'utils', 'des_' + self.args['dataset'] + '.json')
                    cls2des = load_des_file(des_file)
                    cls2txts = {}
                    for i, ci in enumerate(cur_class_names):
                        cls_id = self._known_classes + i
                        if '_' in ci:
                            ct = ' '.join(ci.split('_'))
                        else:
                            ct = ci
                        texts = [t.format(ct) for t in self.prompt_template]
                        texts = self.tokenizer(texts).to(self._device)
                        text_feats = self.raw_clip.encode_text(text=texts, normalize=True).detach()
                        # add class map of description
                        des_text = []
                        if ct.lower() in cls2des:
                            for t in cls2des[ct.lower()]:  # concat with class distribution.
                                des_text.append(t)
                            des_text = self.tokenizer(des_text).to(self._device)
                            des_feat = F.normalize(
                                torch.mean(self.raw_clip.encode_text(des_text, normalize=True), dim=0), dim=-1).detach()
                            des_feat = torch.unsqueeze(des_feat, dim=0)
                            text_feats = (1.0 - self.args['proto_shift']) * text_feats \
                                + self.args['proto_shift'] * des_feat
                            text_feats = F.normalize(text_feats, dim=-1).detach()
                        else:
                            print('no such class:', ci)
                        cls2txts[cls_id] = text_feats
                print('--- build open set class data ---')
                prv_class_names = data_manager.get_classnames(np.arange(self._known_classes))
                data_loader = self.build_outset_inv_data(
                    cls2txts=cls2txts, 
                    transforms=self.train_trfm if self.args['start_block'] == 0 else None, 
                    class_names=prv_class_names,
                    cur_names=cur_class_names
                )
                # train model on openset classes
                print('--- train on open set class data ---')
                self.openset_training(train_loader=data_loader)
        else:
            self._train(train_loader, self.test_loader, tb_logger)
        if self._memory_size > 0 and self._total_classes < data_manager.n_classes:
            # change to inversion
            if bool(self.args['data_stat']):
                # update input feature stats according to real data
                self.inversion_runner.update_input_stat(data_loader=train_loader)
                # update input stat for each layer on raw CLIP model.
                self.openset_inversion_runner.update_input_stat(data_loader=train_loader)
            # update model
            self.old_model = copy.deepcopy(self._network)
            self.old_model.float()
            self.old_model.cuda()
            self.old_model.eval()
            self.old_model.set_is_train(is_train=False)
            freeze_module(module=self.old_model)
            self.inv_model = self.old_model.visual
            self.inversion_runner.update_model(model=self.inv_model)
            # count class distribution.
            if self.args['stat_type'] == 'gmm':
                # add support for GMM feature modelling
                cls2stat, cls2feats = feature_stats.get_gmm_class_distribution(
                    backbone=self.inv_model, data_loader=train_loader,
                    classes=list(range(self._known_classes, self._total_classes)),
                    on_cuda=True, components=self.args['components']
                )
            else:
                cls2stat, cls2feats = feature_stats.get_class_wise_distribution(
                    backbone=self.inv_model, data_loader=train_loader,
                    classes=list(range(self._known_classes, self._total_classes)), on_cuda=True
                )
            # compute feature stats for previous data on raw CLIP
            raw_cls2stat, raw_cls2feats = feature_stats.get_class_wise_distribution(
                backbone=self.raw_clip.visual, data_loader=train_loader,
                classes=list(range(self._known_classes, self._total_classes)), on_cuda=True
            )
            # add momentum center update
            if self.args['stat_mom'] > 0 and self.args['stat_type'] == 'rand':
                self.cls2stat = feature_stats.update_old_centers(
                    cls2stat=self.cls2stat,
                    backbone=self.inv_model,
                    start_block=self.args['start_block'],
                    normalize=False,
                    samples=self._data_memory,
                    labels=self._targets_memory,
                    on_cuda=True,
                    momentum=self.args['stat_mom']
                )
            self.cls2stat.update(cls2stat)
            self.raw_cls2stat.update(raw_cls2stat)
            if self.args['feat_type'] == 'cont':
                self.train_cont_models(
                    cls2feats=cls2feats, start_class=self._known_classes, end_class=self._total_classes, raw=False)
            if 'train_raw_cont' in self.args and bool(self.args['train_raw_cont']):
                self.train_cont_models(
                    cls2feats=raw_cls2feats, start_class=self._known_classes, end_class=self._total_classes, raw=True)
            mem_sample, mem_label = self.build_inv_data(cls2stat=self.cls2stat)
            self._data_memory = mem_sample
            self._targets_memory = mem_label
            # update buffer
            self.buffer = SimpleReplayDataset(
                mem_data=[self._data_memory, self._targets_memory],
                transforms=train_dataset.trsf if self.args['start_block'] == 0 else None
            )
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def save_trained_checkpoint(self):
        print('--- save checkpoint ---')
        save_path = os.path.join(self.local_path, 'model_' + str(self._cur_task) + '.pkl')
        save_model = copy.deepcopy(self._network).eval().cpu()
        torch.save(save_model, save_path)
        stat_path = os.path.join(self.local_path, 'stats_' + str(self._cur_task) + '.pkl')
        with open(stat_path, 'wb') as fw:
            pickle.dump(self.cls2stat, fw)
        raw_stat_path = os.path.join(self.local_path, 'raw_stats_' + str(self._cur_task) + '.pkl')
        with open(raw_stat_path, 'wb') as fw:
            pickle.dump(self.raw_cls2stat, fw)
        # save block input stats
        layer_stats_folder = os.path.join(self.local_path, 'layer_stats_' + str(self._cur_task))
        if not os.path.exists(layer_stats_folder):
            os.makedirs(layer_stats_folder)
        buffer_path = os.path.join(self.local_path, 'buffer')
        for fi in os.listdir(buffer_path):
            if 'input_stats' in fi and fi.endswith('.pkl'):
                src_file = os.path.join(buffer_path, fi)
                dst_file = os.path.join(layer_stats_folder, fi)
                shutil.copy(src_file, dst_file)
        # save contrastive models
        cont_save_file = os.path.join(self.local_path, 'contrastive_models.pkl')
        with open(cont_save_file, 'wb') as fw:
            for mi in self.cont_models:
                mi.cpu()
                pickle.dump(mi, fw)
                mi.to(self._device)
        raw_cont_save_file = os.path.join(self.local_path, 'raw_contrastive_models.pkl')
        with open(raw_cont_save_file, 'wb') as fw:
            for mi in self.raw_cont_models:
                mi.cpu()
                pickle.dump(mi, fw)
                mi.to(self._device)
        # save buffer
        buffer_save_file = os.path.join(self.local_path, 'buffer.pkl')
        with open(buffer_save_file, 'wb') as fw:
            buffer_sps = []
            buffer_labs = self.buffer.mem_labs
            for i in range(self.buffer.mem_cnt):
                buffer_sps.append(self.buffer.mem_sps[i].numpy())
            pickle.dump(buffer_sps, fw)
            pickle.dump(buffer_labs, fw)

    def load_trained_checkpoint(self, transforms):
        print('--- load checkpoint ---')
        # load model
        save_path = os.path.join(self.local_path, 'model_' + str(self.args['ckpt']) + '.pkl')
        save_model = torch.load(save_path)
        self._network = save_model
        self._network.to(self._device)
        # load stats by changing local path
        self.inversion_runner.local_path = os.path.join(
            self.local_path, 'layer_stats_' + str(self.args['ckpt']))
        self.old_model = copy.deepcopy(self._network)
        self.old_model.float()
        self.old_model.cuda()
        self.old_model.eval()
        self.old_model.set_is_train(is_train=False)
        freeze_module(module=self.old_model)
        self.inv_model = self.old_model.visual
        self.inversion_runner.update_model(model=self.inv_model)
        self.inversion_runner.cuda()
        # load contrastive models
        cont_save_file = os.path.join(self.local_path, 'contrastive_models.pkl')
        with open(cont_save_file, 'rb') as fr:
            while True:
                try:
                    mi = pickle.load(fr)
                    self.cont_models.append(mi.to(self._device))
                except EOFError:
                    break
        raw_cont_save_file = os.path.join(self.local_path, 'raw_contrastive_models.pkl')
        if os.path.exists(raw_cont_save_file):
            with open(raw_cont_save_file, 'rb') as fr:
                while True:
                    try:
                        mi = pickle.load(fr)
                        self.raw_cont_models.append(mi.to(self._device))
                    except EOFError:
                        break
        # load buffer
        buffer_save_file = os.path.join(self.local_path, 'buffer.pkl')
        with open(buffer_save_file, 'rb') as fr:
            buffer_sps = pickle.load(fr)
            buffer_labs = pickle.load(fr)
            print('load buffer data:', len(buffer_sps))
            buffer_tensor = []
            for i in range(len(buffer_sps)):
                buffer_tensor.append(torch.tensor(buffer_sps[i], dtype=torch.float32, requires_grad=False))
            self.buffer = SimpleReplayDataset(
                mem_data=[buffer_tensor, buffer_labs],
                transforms=transforms if self.args['start_block'] == 0 else None
            )
        # load class feature stats
        stat_path = os.path.join(self.local_path, 'stats_' + str(self._cur_task) + '.pkl')
        with open(stat_path, 'rb') as fr:
            self.cls2stat = pickle.load(fr)
        raw_stat_path = os.path.join(self.local_path, 'raw_stats_' + str(self._cur_task) + '.pkl')
        with open(raw_stat_path, 'rb') as fr:
            self.raw_cls2stat = pickle.load(fr)

    def _train(self, train_loader, test_loader, tb_logger=None):
        self._network.to(self._device)
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {len(enabled)}")
        # change to MoE
        params = [v for k, v in self._network.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k]
        print('parameters:', len(params))
        self.optimizer = torch.optim.AdamW(params, lr=self.args['init_lr'], weight_decay=self.args['weight_decay'])
        # self.optimizer = build_optimizer(self._network, self.args)
        num_batches = len(train_loader)
        total_iterations = self.args['tuned_epoch'] * num_batches
        self.scheduler = cosine_lr(self.optimizer, self.args['init_lr'], 30, total_iterations)

        self._init_train(train_loader, test_loader, self.optimizer, self.scheduler, tb_logger)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, logger=None):
        # add hard KD loss
        if self.args['hkd_loss'] == 'l1':
            hkd_loss_fn = torch.nn.L1Loss()
        else:
            hkd_loss_fn = torch.nn.MSELoss()
        if self.args['tkd_loss'] == 'l1':
            tkd_loss_fn = torch.nn.L1Loss()
        else:
            tkd_loss_fn = torch.nn.MSELoss()
        # add head finetuining loss
        if 'lambda_ft' in self.args and self.args['lambda_ft'] > 0:
            ft_factor = self.args['lambda_ft']
        else:
            ft_factor = 0
        # changing loss factor by task
        if bool(self.args['change_factor']) and self._known_classes > 0:
            alpha = math.log((self._total_classes - self._known_classes) / 2 + 1, 2)
            beta = math.sqrt(self._known_classes / (self._total_classes - self._known_classes))
            ce_factor = self.args['lambda_ce'] * (1 + 1 / alpha) / beta
            hkd_factor = self.args['lambda_hkd'] * alpha * beta
            tkd_factor = self.args['lambda_tkd'] * alpha * beta
        else:
            ce_factor = 1.0
            hkd_factor = self.args['lambda_hkd']
            tkd_factor = self.args['lambda_tkd']
        cls_count = torch.ones(self._total_classes)
        # add support for start block selection
        self._network.set_is_train(is_train=True)
        if self.args['start_block'] > 0 and self._known_classes > 0:  # only used for replay
            cur_blocks, batch_dim = utils.split_clip_blocks(
                model=self._network.visual, split_cnn=True, normalize=True)
            prv_blocks, batch_dim = utils.split_clip_blocks(model=self.inv_model, split_cnn=True, normalize=True)
            cur_model = torch.nn.Sequential(*cur_blocks[self.args['start_block']:])
            prv_model = torch.nn.Sequential(*prv_blocks[self.args['start_block']:])
        else:
            cur_model = None
            prv_model = None
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        cur_texts = self.tokenizer(self.cur_texts).cuda().to(self._device)
        old_texts = self.tokenizer(self.all_texts[:self._known_classes]).to(self._device)
        all_texts = self.tokenizer(self.all_texts).to(self._device)
        self._network.train()
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for iteration, (_, inputs, targets) in enumerate(train_loader, start=len(train_loader) * epoch):
                start_time = time.time()
                scheduler(iteration)
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                ############################################################
                # self._network.prompt_learner.load_state_dict(self._network.prompt_learner_ema.state_dict())
                ############################################################

                logits, _ = self._network(inputs, cur_texts, 0, is_train=True)
                loss_ce = F.cross_entropy(
                    logits, targets.long() - self._known_classes, label_smoothing=self.args['label_smoothing'])
                loss = ce_factor * loss_ce

                optimizer.zero_grad()
                loss.backward()

                # replay part
                if self._known_classes > 0 and self._memory_size > 0:
                    # hkd loss
                    logit_scale = self._network.logit_scale.exp()
                    if bool(self.args['prv_ft']):
                        old_ft = self.old_model.encode_text(old_texts, normalize=True, is_train=False).detach()
                    else:
                        old_ft = self._network.encode_text(old_texts, normalize=True, is_train=True)
                    mem_sp, mem_lab = self.buffer.get_batch(size=inputs.shape[0])
                    mem_sp, mem_lab = mem_sp.to(self._device), mem_lab.to(self._device)
                    if self.args['start_block'] == 0:  # consider MoE case
                        if bool(self.args['prv_ft']):
                            old_fi = self._network.encode_image(mem_sp, normalize=True, is_train=True)
                            old_logit = logit_scale * old_fi @ old_ft.t()
                        else:
                            old_logit, _ = self._network(mem_sp, old_texts, 0, is_train=True)
                        teacher_fi, _ = self.inv_model(mem_sp)
                        teacher_fi = teacher_fi.detach()
                        teacher_fi = teacher_fi / teacher_fi.norm(dim=-1, keepdim=True)
                    else:
                        old_fi = cur_model(mem_sp)  # add support for is_train
                        teacher_fi = prv_model(mem_sp).detach()
                        old_logit = logit_scale * old_fi @ old_ft.t()
                    teacher_logit = logit_scale * teacher_fi @ old_ft.t()
                    loss_hkd = hkd_factor * hkd_loss_fn(old_logit, teacher_logit)
                    loss_hkd.backward()
                    all_targets = torch.cat([targets, mem_lab], dim=0)
                    indices, counts = all_targets.cpu().unique(return_counts=True)
                    cls_count[indices] += counts
                    # add text replay.
                    student_ft = self._network.encode_text(old_texts, normalize=True, is_train=True)
                    teacher_ft = self.old_model.encode_text(old_texts, normalize=True, is_train=False).detach()
                    loss_tkd = tkd_factor * tkd_loss_fn(student_ft, teacher_ft)
                    loss_tkd.backward()
                    if 'lambda_ft' in self.args and self.args['lambda_ft'] > 0:
                        # add classification head finetuning
                        if 'ft_balance' in self.args and bool(self.args['ft_balance']):
                            class_weights = cls_count.sum() / cls_count.clamp(min=1)  # add weight of classes
                            class_weights = class_weights.div(class_weights.min())
                            class_weights = class_weights.to(self._device)
                        else:
                            class_weights = torch.ones(self._total_classes).to(self._device)
                        ft_old_fi = cur_model(mem_sp).detach()
                        all_ft = self._network.encode_text(all_texts, normalize=True, is_train=True)
                        old_logit = logit_scale * ft_old_fi @ all_ft.t()
                        loss_ft_old = ft_factor * torch.nn.functional.cross_entropy(old_logit, mem_lab, class_weights)
                        loss_ft_old.backward()
                        ft_new_fi = self._network.encode_image(inputs, normalize=True, is_train=True).detach()
                        all_ft = self._network.encode_text(all_texts, normalize=True, is_train=True)
                        new_logit = logit_scale * ft_new_fi @ all_ft.t()
                        loss_ft_new = ft_factor * torch.nn.functional.cross_entropy(new_logit, targets, class_weights)
                        loss_ft_new.backward()
                    if iteration % 100 == 0:
                        print('losses at step', iteration, 'hkd loss:', loss_hkd.item(), 'tkd loss:', loss_tkd.item())

                if logger:
                    total_iterations = len(train_loader) * self.epochs
                    logger.add_scalar("loss/train", loss, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss/ce", loss_ce, iteration + self._cur_task * total_iterations)
                    logger.add_scalar('Lr', optimizer.param_groups[0]['lr'],
                                      iteration + self._cur_task * total_iterations)

                optimizer.step()
                ############################################################
                # ema_update(self._network.prompt_learner, self._network.prompt_learner_ema, ema_decay=0.999)
                ############################################################

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                end_time = time.time()
                elapsed_time = end_time - start_time

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )

            logging.info(info)
        self._network.eval()
        self._network.set_is_train(is_train=None)
        if self._known_classes > 0 and self.args['finetune_epoch'] > 0:
            self.finetune_classifier(train_loader=train_loader, class_count=cls_count)

    def openset_training(self, train_loader):
        # add support for open set class training
        self._network.to(self._device)
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {len(enabled)}")
        # change to MoE
        params = [v for k, v in self._network.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k]
        print('parameters:', len(params))
        optimizer = torch.optim.AdamW(params, lr=self.args['open_lr'], weight_decay=self.args['weight_decay'])
        # self.optimizer = build_optimizer(self._network, self.args)
        num_batches = len(train_loader)
        total_iterations = 3 * num_batches
        scheduler = cosine_lr(optimizer, self.args['open_lr'], 30, total_iterations)
        # add hard KD loss
        if self.args['hkd_loss'] == 'l1':
            hkd_loss_fn = torch.nn.L1Loss()
        else:
            hkd_loss_fn = torch.nn.MSELoss()
        if self.args['tkd_loss'] == 'l1':
            tkd_loss_fn = torch.nn.L1Loss()
        else:
            tkd_loss_fn = torch.nn.MSELoss()
        # add head finetuining loss
        if 'openset_ft' in self.args and self.args['openset_ft'] > 0:
            ft_factor = self.args['openset_ft']
        else:
            ft_factor = 0
        # changing loss factor by task
        if bool(self.args['change_factor']) and self._known_classes > 0:
            alpha = math.log((self._total_classes - self._known_classes) / 2 + 1, 2)
            beta = math.sqrt(self._known_classes / (self._total_classes - self._known_classes))
            ce_factor = self.args['openset_ce'] * (1 + 1 / alpha) / beta
            hkd_factor = self.args['openset_hkd'] * alpha * beta
            tkd_factor = self.args['openset_tkd'] * alpha * beta
        else:
            ce_factor = 1.0
            hkd_factor = self.args['openset_hkd']
            tkd_factor = self.args['openset_tkd']
        cls_count = torch.ones(self._total_classes)
        # add support for start block selection
        self._network.set_is_train(is_train=True)
        if self.args['start_block'] > 0 and self._known_classes > 0:  # only used for replay
            cur_blocks, batch_dim = utils.split_clip_blocks(
                model=self._network.visual, split_cnn=True, normalize=True)
            prv_blocks, batch_dim = utils.split_clip_blocks(model=self.inv_model, split_cnn=True, normalize=True)
            cur_model = torch.nn.Sequential(*cur_blocks[self.args['start_block']:])
            prv_model = torch.nn.Sequential(*prv_blocks[self.args['start_block']:])
        else:
            cur_model = None
            prv_model = None
        prog_bar = tqdm(range(self.args['openset_epoch']))
        cur_texts = self.tokenizer(self.cur_texts).cuda().to(self._device)
        old_texts = self.tokenizer(self.all_texts[:self._known_classes]).to(self._device)
        all_texts = self.tokenizer(self.all_texts).to(self._device)
        self._network.train()
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            self._network.set_is_train(is_train=None)
            cnn_accy, _ = self.eval_task()  # set_istrain before evaluation
            print(cnn_accy["grouped"])
            self._network.set_is_train(is_train=True)
            for iteration, (inputs, targets) in enumerate(train_loader, start=len(train_loader) * epoch):
                start_time = time.time()
                scheduler(iteration)
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self.args['start_block'] == 0:  # add support for start block
                    logits, _ = self._network(inputs, cur_texts, 0, is_train=True)
                else:
                    cur_fi = cur_model(inputs)
                    cur_ft = self._network.encode_text(cur_texts, normalize=True, is_train=True)
                    logit_scale = self._network.logit_scale.exp()
                    logits = logit_scale * cur_fi @ cur_ft.t()
                loss_ce = F.cross_entropy(
                    logits, targets.long() - self._known_classes, label_smoothing=self.args['label_smoothing'])
                loss = ce_factor * loss_ce
                optimizer.zero_grad()
                loss.backward()

                # replay part
                if self._known_classes > 0 and self._memory_size > 0:
                    # hkd loss
                    logit_scale = self._network.logit_scale.exp()
                    if bool(self.args['prv_ft']):
                        old_ft = self.old_model.encode_text(old_texts, normalize=True, is_train=False).detach()
                    else:
                        old_ft = self._network.encode_text(old_texts, normalize=True, is_train=True)
                    mem_sp, mem_lab = self.buffer.get_batch(size=inputs.shape[0])
                    mem_sp, mem_lab = mem_sp.to(self._device), mem_lab.to(self._device)
                    if self.args['start_block'] == 0:  # consider MoE case
                        if bool(self.args['prv_ft']):
                            old_fi = self._network.encode_image(mem_sp, normalize=True, is_train=True)
                            old_logit = logit_scale * old_fi @ old_ft.t()
                        else:
                            old_logit, _ = self._network(mem_sp, old_texts, 0, is_train=True)
                        teacher_fi, _ = self.inv_model(mem_sp)
                        teacher_fi = teacher_fi.detach()
                        teacher_fi = teacher_fi / teacher_fi.norm(dim=-1, keepdim=True)
                    else:
                        old_fi = cur_model(mem_sp)  # add support for is_train
                        teacher_fi = prv_model(mem_sp).detach()
                        old_logit = logit_scale * old_fi @ old_ft.t()
                    teacher_logit = logit_scale * teacher_fi @ old_ft.t()
                    loss_hkd = hkd_factor * hkd_loss_fn(old_logit, teacher_logit)
                    loss_hkd.backward()
                    all_targets = torch.cat([targets, mem_lab], dim=0)
                    indices, counts = all_targets.cpu().unique(return_counts=True)
                    cls_count[indices] += counts
                    # add text replay.
                    student_ft = self._network.encode_text(old_texts, normalize=True, is_train=True)
                    teacher_ft = self.old_model.encode_text(old_texts, normalize=True, is_train=False).detach()
                    loss_tkd = tkd_factor * tkd_loss_fn(student_ft, teacher_ft)
                    loss_tkd.backward()
                    if 'openset_ft' in self.args and self.args['openset_ft'] > 0:
                        # add classification head finetuning
                        if 'ft_balance' in self.args and bool(self.args['ft_balance']):
                            class_weights = cls_count.sum() / cls_count.clamp(min=1)  # add weight of classes
                            class_weights = class_weights.div(class_weights.min())
                            class_weights = class_weights.to(self._device)
                        else:
                            class_weights = torch.ones(self._total_classes).to(self._device)
                        ft_old_fi = cur_model(mem_sp).detach()
                        all_ft = self._network.encode_text(all_texts, normalize=True, is_train=True)
                        old_logit = logit_scale * ft_old_fi @ all_ft.t()
                        loss_ft_old = ft_factor * torch.nn.functional.cross_entropy(old_logit, mem_lab, class_weights)
                        loss_ft_old.backward()
                        ft_new_fi = cur_model(inputs).detach()
                        all_ft = self._network.encode_text(all_texts, normalize=True, is_train=True)
                        new_logit = logit_scale * ft_new_fi @ all_ft.t()
                        loss_ft_new = ft_factor * torch.nn.functional.cross_entropy(new_logit, targets, class_weights)
                        loss_ft_new.backward()
                    if iteration % 100 == 0:
                        print('losses at step', iteration, 'hkd loss:', loss_hkd.item(), 'tkd loss:', loss_tkd.item(),
                              'ce loss:', loss_ce.item())

                optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds + self._known_classes)).cpu().sum()
                total += len(targets)

                end_time = time.time()
                elapsed_time = end_time - start_time

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )

            logging.info(info)
        self._network.eval()
        self._network.set_is_train(is_train=None)
        if self._known_classes > 0 and self.args['finetune_epoch'] > 0:
            self.finetune_classifier(train_loader=train_loader, class_count=cls_count)

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.visual(_inputs.to(self._device))
                    )
                else:
                    f, _ = self._network.visual(_inputs.to(self._device))
                    _vectors = tensor2numpy(f)

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def eval_task(self, tb_logger=None, network=None):
        y_pred, y_true = self._eval_cnn(self.test_loader, network=network)
        cnn_accy = self._evaluate(y_pred, y_true)
        if tb_logger:
            tb_logger.add_scalar("ACC/acc", cnn_accy['grouped']['total'], self._cur_task)

        f_y_pred, f_y_true = self._eval_cnn(self.full_test_loader)
        f_cnn_accy = self._evaluate(f_y_pred, f_y_true)
        logging.info("Full CNN: {}".format(f_cnn_accy["grouped"]))

        nme_accy = None

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader, network=None):
        if network is None:
            self._network.eval()
            network = self._network
        y_pred, y_true = [], []
        texts = self.tokenizer(self.all_texts).cuda().to(self._device)  # change to MoE
        for _, (_, inputs, targets) in enumerate(loader):
            start_time = time.time()
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits, _ = network(inputs, texts, 0, is_train=False)
            predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            end_time = time.time()
            elapsed_time = end_time - start_time

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def finetune_classifier(self, train_loader, class_count):
        # only tune text encoder
        print('--- start tuning text encoder ---')
        self._network.train()
        self._network.visual.eval()
        all_texts = self.tokenizer(self.all_texts).to(self._device)
        teacher_ft = self.old_model.encode_text(all_texts, normalize=True, is_train=False).detach()
        params = [v for k, v in self._network.named_parameters()
                  if ("adaptmlp" in k or "router" in k or "noise" in k) and 'visual' not in k]
        print('parameters:', len(params))
        opt = torch.optim.AdamW(params=params, lr=self.args['finetune_lr'])
        loss_factor = 0.5
        hkd_factor = self.args['lambda_hkd'] * self._known_classes / (self._total_classes - self._known_classes)
        hkd_loss_fn = torch.nn.L1Loss()
        logit_scale = self._network.logit_scale.exp()
        self._network.transformer.set_is_train(is_train=True)
        if self.args['start_block'] > 0 and self._known_classes > 0:  # only used for replay
            cur_blocks, batch_dim = utils.split_clip_blocks(
                model=self._network.visual, split_cnn=True, normalize=True)
            cur_model = torch.nn.Sequential(*cur_blocks[self.args['start_block']:])
            prv_blocks, batch_dim = utils.split_clip_blocks(model=self.inv_model, split_cnn=True, normalize=True)
            prv_model = torch.nn.Sequential(*prv_blocks[self.args['start_block']:])
        else:
            cur_model = None
            prv_model = None
        for e in range(self.args['finetune_epoch']):
            step = 0
            for data in train_loader:
                class_weights = class_count.sum() / class_count.clamp(min=1)
                class_weights = class_weights.div(class_weights.min())
                class_weights = class_weights.to(self._device)
                opt.zero_grad()
                _, sps, labs = data
                sps, labs = sps.to(self._device), labs.to(self._device)
                inv_sps, inv_labs = self.buffer.get_batch(size=sps.shape[0])
                inv_sps, inv_labs = inv_sps.to(self._device), inv_labs.to(self._device)
                cur_logits, _ = self._network(sps, all_texts, 0, is_train=True)
                cur_loss = loss_factor * torch.nn.functional.cross_entropy(cur_logits, labs, class_weights)
                cur_loss.backward()
                all_ft = self._network.encode_text(all_texts, normalize=True, is_train=True)
                if self.args['start_block'] == 0:
                    inv_fi = self._network.encode_image(inv_sps, normalize=True, is_train=False).detach()
                    inv_logits = logit_scale * inv_fi @ all_ft.t()
                else:
                    inv_fi = cur_model(inv_sps).detach()
                    inv_logits = logit_scale * inv_fi @ all_ft.t()
                target_all = torch.cat([labs, inv_labs])
                prv_loss = loss_factor * torch.nn.functional.cross_entropy(inv_logits, inv_labs, class_weights)
                prv_loss.backward()
                if step % 100 == 0:
                    print('finetuning loss at step', step, 'is:', cur_loss.item(), prv_loss.item())
                all_ft = self._network.encode_text(all_texts, normalize=True, is_train=True)
                if self.args['start_block'] == 0:
                    inv_fi = self._network.encode_image(inv_sps, normalize=True, is_train=False).detach()
                    inv_logits = logit_scale * inv_fi @ all_ft.t()
                    teacher_logit = self.old_model(
                        inv_sps, all_texts, 0, is_train=False)[:, :self._known_classes].detach()
                else:
                    inv_fi = cur_model(inv_sps).detach()
                    inv_logits = logit_scale * inv_fi @ all_ft.t()
                    teacher_fi = prv_model(inv_sps).detach()
                    teacher_logit = logit_scale * teacher_fi @ teacher_ft.t()
                    teacher_logit = teacher_logit[:, :self._known_classes].detach()
                l_hkd = hkd_loss_fn(inv_logits[:, :self._known_classes], teacher_logit)
                loss_hkd = hkd_factor * l_hkd
                loss_hkd.backward()
                opt.step()
                if step % 100 == 0:
                    print('finetuning hkd loss at step', step, 'is:', loss_hkd.item())
                indices, counts = target_all.cpu().unique(return_counts=True)
                class_count[indices] += counts
                step += 1
        self._network.set_is_train(is_train=None)
        self._network.eval()

    def build_inv_data(self, cls2stat):
        print('number of inversion class:', len(cls2stat))
        train_samples = []
        train_labels = []
        target_feats = []
        for ci in cls2stat.keys():
            # random sample text features
            stat = cls2stat[ci]
            feat_size = self.args['memory_per_class']
            if self.args['feat_type'] == 'rand':
                if self.args['stat_type'] == 'gmm':
                    cls_feats, _ = stat.sample(n_samples=feat_size)
                    cls_feats = torch.tensor(cls_feats, dtype=torch.float32, requires_grad=False)
                else:
                    cls_mean, cls_std = cls2stat[ci]
                    eps = torch.randn(
                        size=[feat_size, cls_mean.shape[1]],
                        dtype=torch.float32, requires_grad=False
                    )
                    cls_feats = eps * cls_std + cls_mean
            elif self.args['feat_type'] == 'cont':
                # add options of contrastive selection
                if 'boost_rate' in self.args:  # add support for boosting selection number and random select
                    boost_rate = self.args['boost_rate']
                else:
                    boost_rate = 1
                batch_size = max(int(feat_size * boost_rate // self.args['cont_step']), 1)
                cls_feats = cont_model.contrastive_selection(
                    cont_model=self.cont_models[ci],
                    stats=stat,
                    samples=feat_size * boost_rate,
                    batch_size=batch_size,
                    slt_rate=self.args['cont_slt_rate'],
                    feat_dim=self._network.feature_dim,
                    on_cuda=True,
                    tau=1.0
                )
                if boost_rate > 1:
                    slt_ids = random.sample(list(range(cls_feats.shape[0])), feat_size)
                    slt_ids = np.array(slt_ids)
                    cls_feats = cls_feats[slt_ids, :]
            else:
                raise ValueError('Invalid feat type', self.args['feat_type'])
            train_labels.append(np.full(feat_size, ci))
            target_feats.append(cls_feats)
        train_labels = np.concatenate(train_labels, axis=0)
        target_feats = torch.cat(target_feats, dim=0)
        rand_ord = list(range(target_feats.shape[0]))
        random.shuffle(rand_ord)
        rand_ord = np.array(rand_ord)
        target_feats = target_feats[rand_ord, :]
        train_labels = train_labels[rand_ord]
        # do model inversion
        sample_count = 0
        size_change = None
        if len(self.args['size_change']) > 0:
            size_change = [int(si) for si in self.args['size_change'].split(',')]
        if len(self.args['layer_size_change']) > 0:
            layer_iters = [int(si) for si in self.args['layer_size_change'].split(',')]
            first_layer_param = {
                'iters': layer_iters[-1],
                'size_change': layer_iters[:-1],
                'alpha_pr': self.args['l_pr'],
                'l_aug': bool(self.args['l_aug']),
                'l_lr': self.args['l_lr']
            }
        else:
            first_layer_param = {
                'iters': None,
                'size_change': None,
                'alpha_pr': self.args['l_pr'],
                'l_aug': bool(self.args['l_aug']),
                'l_lr': self.args['l_lr']
            }
        while sample_count < target_feats.shape[0]:
            bs = min(self.args['gen_batch_size'], target_feats.shape[0] - sample_count)
            layer_batch = min(self.args['layer_batch'], target_feats.shape[0] - sample_count)
            gen_img = self.inversion_runner.layer_wise_inversion_for_cl(
                batch_size=bs,
                target_feats=target_feats[sample_count:sample_count + layer_batch],
                size_change=size_change,
                milestones=[int(mi) for mi in self.args['milestones'].split(',')],
                finetune_iters=self.args['tune_steps'],
                finetune_lr=self.args['tune_lr'],
                iters=self.args['train_steps'],
                return_best=bool(self.args['return_best']),
                first_layer_param=first_layer_param,
                search_param=bool(self.args['search_params']),
                verbose=True,
                gradual_rf=False
            )
            gen_img = gen_img.detach().cpu()
            sample_count += gen_img.shape[0]
            train_samples.append(gen_img)
            print('finish generating samples:', sample_count)
        train_samples = torch.cat(train_samples, dim=0)
        pil_samples = []
        if self.args['start_block'] == 0:
            to_pil = torchvision.transforms.ToPILImage()
            for i in range(train_samples.shape[0]):
                pil_samples.append(to_pil(train_samples[i, :]))
        else:
            for i in range(train_samples.shape[0]):
                pil_samples.append(train_samples[i, :])
        return pil_samples, train_labels

    def train_cont_models(self, cls2feats, start_class, end_class, raw=False):
        model_params = {
            'blocks': self.args['cont_blocks'],
            'lr': self.args['cont_lr'],
            'epoch': self.args['cont_epoch'],
            'tau': 1.0
        }
        for ci in range(start_class, end_class):
            print('train contrastive model for class:', ci)
            # prepare file
            temp_file = os.path.join(self.local_path, 'cont_train_file.pkl')
            with open(temp_file, 'wb') as fw:
                for i in range(len(cls2feats[ci])):
                    pickle.dump([cls2feats[ci][i], ci], fw)
            # train cont model
            trained_model = cont_model.train_contrastive_model(
                data_file=temp_file, use_cuda=True, model_params=model_params, act='leaky', verbose=False)
            # update cont model
            if raw:
                self.raw_cont_models.append(trained_model)
            else:
                self.cont_models.append(trained_model)
            os.remove(temp_file)

    def build_outset_inv_data(self, cls2txts, transforms, class_names, cur_names):
        # prepare for labels and features.
        target_feats = []
        targets = []
        if self.args['open_feat_mode'] == 'proto_map':
            cls2feats = self.prototype_mapping(class_names=class_names, cur_names=cur_names)
        else:
            cls2feats = None
        for ci in cls2txts.keys():
            feat_size = self.args['openset_per_class']
            text_feats = cls2txts[ci]
            if self.args['open_feat_mode'] == 'gaussian':  # make stat based on text features.
                mean = torch.mean(text_feats, dim=0, keepdim=True)
                std = torch.std(text_feats, dim=0, keepdim=True)
                eps = torch.randn(size=[feat_size, mean.shape[1]], dtype=torch.float32, requires_grad=False).cuda()
                feats = eps * std + mean
                target_feats.append(feats)
            elif self.args['open_feat_mode'] == 'center':  # only use center
                center = torch.mean(text_feats, dim=0, keepdim=True)
                feats = []
                for j in range(feat_size):
                    feats.append(center)
                target_feats.append(torch.cat(feats, dim=0))
            elif self.args['open_feat_mode'] == 'proto_map':
                target_feats.append(torch.tensor(cls2feats[ci], dtype=torch.float32, requires_grad=False))
            else:  # select from text features
                if 'cont' in self.args['open_feat_mode']:
                    feats = self.contrastive_text_selection(txt_feats=text_feats, feat_size=feat_size)
                else:
                    rand_ids = np.random.randint(low=0, high=text_feats.shape[0], size=feat_size)
                    feats = text_feats[rand_ids, :]
                target_feats.append(feats)
            targets.append(np.full(feat_size, ci))
        targets = np.concatenate(targets, axis=0)
        target_feats = torch.cat(target_feats, dim=0)
        target_feats = F.normalize(target_feats, dim=-1).detach()
        rand_ord = list(range(target_feats.shape[0]))
        random.shuffle(rand_ord)
        rand_ord = np.array(rand_ord)
        target_feats = target_feats[rand_ord, :]
        targets = targets[rand_ord]
        # do model inversion.
        sample_count = 0
        size_change = None
        train_samples = []
        if len(self.args['size_change']) > 0:
            size_change = [int(si) for si in self.args['size_change'].split(',')]
        if len(self.args['layer_size_change']) > 0:
            layer_iters = [int(si) for si in self.args['layer_size_change'].split(',')]
            first_layer_param = {
                'iters': layer_iters[-1],
                'size_change': layer_iters[:-1],
                'alpha_pr': self.args['l_pr'],
                'l_aug': bool(self.args['l_aug']),
                'l_lr': self.args['l_lr']
            }
        else:
            first_layer_param = {
                'iters': None,
                'size_change': None,
                'alpha_pr': self.args['l_pr'],
                'l_aug': bool(self.args['l_aug']),
                'l_lr': self.args['l_lr']
            }
        while sample_count < target_feats.shape[0]:
            bs = min(self.args['gen_batch_size'], target_feats.shape[0] - sample_count)
            layer_batch = min(self.args['layer_batch'], target_feats.shape[0] - sample_count)
            gen_img = self.openset_inversion_runner.layer_wise_inversion_for_cl(
                batch_size=bs,
                target_feats=target_feats[sample_count:sample_count + layer_batch],
                size_change=size_change,
                milestones=[int(mi) for mi in self.args['milestones'].split(',')],
                finetune_iters=self.args['tune_steps'],
                finetune_lr=self.args['tune_lr'],
                iters=self.args['train_steps'],
                return_best=bool(self.args['return_best']),
                first_layer_param=first_layer_param,
                search_param=bool(self.args['search_params']),
                verbose=True,
                gradual_rf=False
            )
            gen_img = gen_img.detach().cpu()
            sample_count += gen_img.shape[0]
            train_samples.append(gen_img)
            print('finish generating samples:', sample_count)
        train_samples = torch.cat(train_samples, dim=0)
        # collect data and make data loader
        pil_samples = []
        if self.args['start_block'] == 0:
            to_pil = torchvision.transforms.ToPILImage()
            for i in range(train_samples.shape[0]):
                pil_samples.append(to_pil(train_samples[i, :]))
        else:
            for i in range(train_samples.shape[0]):
                pil_samples.append(train_samples[i, :])
        train_dataset = RandomDataset(samples=pil_samples, labels=targets, transforms=transforms)
        train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True, drop_last=False)
        return train_loader

    def prototype_mapping(self, class_names, cur_names):
        des_file = os.path.join(os.getcwd(), 'utils', 'des_' + self.args['dataset'] + '.json')
        cls2des = load_des_file(des_file)
        raw_text_feats = []
        for i, ci in enumerate(class_names):
            if i == self._known_classes:
                break
            texts = [t.format(ci) for t in self.prompt_template]
            texts = self.tokenizer(texts).to(self._device)
            text_feat = F.normalize(torch.mean(self.raw_clip.encode_text(texts, normalize=True), dim=0), dim=-1)
            text_feat = text_feat.detach()
            if 'text_aug_mode' in self.args and self.args['text_aug_mode'] == 'proto_map':
                des_text = []
                if '_' in ci:
                    ct = ' '.join(ci.split('_'))
                else:
                    ct = ci
                if ct.lower() in cls2des:
                    for t in cls2des[ct.lower()]:  # concat with class distribution.
                        des_text.append(t)
                    des_text = self.tokenizer(des_text).to(self._device)
                    des_feat = F.normalize(torch.mean(
                        self.raw_clip.encode_text(des_text, normalize=True), dim=0), dim=-1)
                    des_feat = des_feat.detach()
                    text_feat = (1.0 - self.args['proto_shift']) * text_feat + self.args['proto_shift'] * des_feat
                    text_feat = F.normalize(text_feat, dim=-1).detach()
                else:
                    print('no such class', ct)
            raw_text_feats.append(text_feat)
        raw_text_feats = torch.stack(raw_text_feats, dim=0)
        raw_image_feats = {}
        for i, ci in enumerate(cur_names):
            lab = i + self._known_classes
            texts = [t.format(ci) for t in self.prompt_template]
            texts = self.tokenizer(texts).to(self._device)
            text_feat = F.normalize(torch.mean(self.raw_clip.encode_text(texts, normalize=True), dim=0), dim=-1)
            text_feat = torch.unsqueeze(text_feat, dim=0)
            text_feat = text_feat.detach()
            if 'text_aug_mode' in self.args and self.args['text_aug_mode'] == 'proto_map':
                des_text = []
                if '_' in ci:
                    ct = ' '.join(ci.split('_'))
                else:
                    ct = ci
                if ct.lower() in cls2des:
                    for t in cls2des[ct.lower()]:  # concat with class distribution.
                        des_text.append(t)
                    des_text = self.tokenizer(des_text).to(self._device)
                    des_feat = F.normalize(
                        torch.mean(self.raw_clip.encode_text(des_text, normalize=True), dim=0), dim=-1)
                    des_feat = des_feat.detach()
                    text_feat = (1.0 - self.args['proto_shift']) * text_feat + self.args['proto_shift'] * des_feat
                    text_feat = F.normalize(text_feat, dim=-1).detach()
                else:
                    print('no such class', ct)
            # select nearest class
            cos_sim = torch.sum(raw_text_feats * text_feat, dim=1).detach().cpu().numpy()
            sim_order = np.flip(np.argsort(cos_sim))
            all_lab_feats = []
            for prv_ci in range(self.args['proto_class']):  # support for multiple classes
                cand_ci = int(sim_order[prv_ci])
                # sample image features
                cls_mean, cls_std = self.raw_cls2stat[cand_ci]
                if 'open_feat_type' in self.args and self.args['open_feat_type'] == 'cont':
                    # add options of contrastive selection
                    feat_size = self.args['openset_per_class']
                    if 'boost_rate' in self.args:  # add support for boosting selection number and random select
                        boost_rate = self.args['boost_rate']
                    else:
                        boost_rate = 1
                    batch_size = max(int(feat_size * boost_rate // self.args['cont_step']), 1)
                    cls_feats = cont_model.contrastive_selection(
                        cont_model=self.raw_cont_models[cand_ci],
                        stats=self.raw_cls2stat[cand_ci],
                        samples=feat_size * boost_rate,
                        batch_size=batch_size,
                        slt_rate=self.args['cont_slt_rate'],
                        feat_dim=self._network.feature_dim,
                        on_cuda=True,
                        tau=1.0
                    )
                    if boost_rate > 1:
                        slt_ids = random.sample(list(range(cls_feats.shape[0])), feat_size)
                        slt_ids = np.array(slt_ids)
                        cls_feats = cls_feats[slt_ids, :]
                else:
                    eps = torch.randn(
                        size=[self.args['openset_per_class'], cls_mean.shape[1]],
                        dtype=torch.float32, requires_grad=False
                    )
                    cls_feats = eps * cls_std + cls_mean
                cls_feats = F.normalize(cls_feats, dim=-1)  # normalize due to CLIP method
                cls_feats = cls_feats.detach().cpu().numpy()
                # map features
                old_proto = raw_text_feats[cand_ci, :].cpu().numpy()
                new_proto = np.squeeze(text_feat.cpu().numpy())
                mapped_feats = []
                for j in range(cls_feats.shape[0]):
                    if 'map_mode' in self.args and self.args['map_mode'] == 'txt_img':
                        mapped = rotate_features(
                            pro_a=np.squeeze(F.normalize(cls_mean, dim=-1).detach().cpu().numpy()),
                            pro_b=new_proto, x=cls_feats[j, :]
                        )
                    else:
                        mapped = rotate_features(pro_a=old_proto, pro_b=new_proto, x=cls_feats[j, :])
                    mapped_feats.append(mapped)
                mapped_feats = np.stack(mapped_feats, axis=0)
                all_lab_feats.append(mapped_feats)
            # average over mapped features
            all_lab_feats = np.stack(all_lab_feats, axis=0)
            sub_lab_feats = np.mean(all_lab_feats, axis=0)
            # normalization for candidate features
            sub_lab_feats = sub_lab_feats / np.linalg.norm(sub_lab_feats, ord=2, axis=-1, keepdims=True)
            if 'text_aug_mode' in self.args and self.args['text_aug_mode'] == 'proto_map':
                aug_feats = []
                for ti in range(self.args['openset_per_class']):
                    aug_feat = (1.0 - self.args['shift_factor']) * sub_lab_feats[ti, :] \
                        + self.args['shift_factor'] * np.squeeze(text_feat.cpu().numpy())
                    aug_feat = aug_feat / np.linalg.norm(aug_feat, 2)
                    aug_feats.append(aug_feat)
                raw_image_feats[lab] = numpy.stack(aug_feats, axis=0)
            else:
                raw_image_feats[lab] = sub_lab_feats
        return raw_image_feats

    def text_augment(self, cur_names, aug_method):
        # load class description file
        des_file = os.path.join(os.getcwd(), 'utils', 'des_' + self.args['dataset'] + '.json')
        cls2des = load_des_file(des_file)
        cls2txt_feats = {}
        with torch.no_grad():
            for i, ci in enumerate(cur_names):
                lab = i + self._known_classes
                if '_' in ci:
                    ct = ' '.join(ci.split('_'))
                else:
                    ct = ci
                if ct.lower() in cls2des:
                    texts = []
                    if aug_method == 'append_des':
                        for t in self.prompt_template:
                            txt = t.format(ct)
                            for di in cls2des[ct.lower()]:
                                aug_txt = txt + ' ' + di
                                texts.append(aug_txt)
                        aug_texts = self.tokenizer(texts).to(self._device)
                        text_feats = self.compute_text_feats_batch(in_texts=aug_texts)
                    elif aug_method == 'shift_comb':
                        proto_texts = []
                        for t in self.prompt_template:
                            txt = t.format(ct)
                            proto_texts.append(txt)
                        proto_in = self.tokenizer(proto_texts).to(self._device)
                        proto = F.normalize(
                            torch.mean(self.raw_clip.encode_text(proto_in, normalize=True), dim=0),
                            dim=-1
                        ).detach()
                        des_text = []
                        for t in cls2des[ct.lower()]:  # concat with class distribution.
                            des_text.append(t)
                        des_text = self.tokenizer(des_text).to(self._device)
                        des_feat = F.normalize(
                            torch.mean(self.raw_clip.encode_text(des_text, normalize=True), dim=0), dim=-1).detach()
                        proto = (1.0 - self.args['proto_shift']) * proto + self.args['proto_shift'] * des_feat
                        proto = F.normalize(proto, dim=-1).detach()
                        # append random subset of descriptions
                        text_feats = []
                        for j in range(self.args['openset_per_class'] * self.args['boost_rate']):
                            des_size = np.random.randint(low=1, high=len(cls2des[ct.lower()]))
                            des_inds = random.sample(list(range(len(cls2des[ct.lower()]))), int(des_size))
                            sub_des = []
                            for ind in des_inds:
                                sub_des.append(cls2des[ct.lower()][ind])
                            sub_des_text = self.tokenizer(sub_des).to(self._device)
                            sub_des_feat = F.normalize(
                                torch.mean(self.raw_clip.encode_text(sub_des_text, normalize=True), dim=0), dim=-1
                            ).detach()
                            text_feat = (1.0 - self.args['shift_factor']) * proto \
                                + self.args['shift_factor'] * sub_des_feat
                            text_feats.append(text_feat)
                        text_feats = torch.stack(text_feats, dim=0)
                    else:
                        text_feats = []
                        for t in self.prompt_template:
                            txt = t.format(ct)
                            for di in cls2des[ct.lower()]:
                                temp_in = self.tokenizer([txt, di]).to(self._device)
                                temp_feat = self.raw_clip.encode_text(temp_in, normalize=True).detach()
                                temp_feat = (1.0 - self.args['shift_factor']) * temp_feat[0] + \
                                    self.args['shift_factor'] * temp_feat[1]
                                temp_feat = F.normalize(temp_feat, dim=-1).detach().cpu()
                                text_feats.append(temp_feat)
                        text_feats = torch.stack(text_feats, dim=0)
                    cls2txt_feats[lab] = text_feats
                    print('augment', text_feats.shape[0], 'for class', lab)
                else:
                    print('no such class', ct)
        # self.compute_feature_loss(cls2txt_feat=cls2txt_feats)
        return cls2txt_feats

    def compute_feature_loss(self, cls2txt_feat):
        cur_texts = self.tokenizer(self.cur_texts).cuda().to(self._device)
        cur_ft = self._network.encode_text(cur_texts, normalize=True, is_train=False).detach()
        logit_scale = self._network.logit_scale.exp()
        with torch.no_grad():
            for lab in cls2txt_feat.keys():
                txt_feats = cls2txt_feat[lab].to(self._device)
                label = (torch.ones(txt_feats.shape[0], requires_grad=False) * lab).long()
                label = label.to(self._device)
                logit = logit_scale * txt_feats @ cur_ft.t()
                loss = F.cross_entropy(
                    logit, label - self._known_classes, label_smoothing=self.args['label_smoothing'])
                print('loss for class', lab, 'is:', loss.item())

    def compute_text_feats_batch(self, in_texts):
        in_texts = in_texts.to(self._device)
        steps = int(in_texts.shape[0] // 100)
        if in_texts.shape[0] % 100 != 0:
            steps += 1
        txt_out = []
        for i in range(steps):
            start = i * 100
            end = min(in_texts.shape[0], (i + 1) * 100)
            inputs = in_texts[start:end, :]
            text_feat = self.raw_clip.encode_text(inputs, normalize=True).detach()
            txt_out.append(text_feat)
        txt_out = torch.cat(txt_out, dim=0)
        return txt_out

    def contrastive_text_selection(self, txt_feats, feat_size):
        # selecting diversion subset for inversion
        cont_loss_fn = cont_model.SelectContrastiveLoss(tau=1.0)
        all_feats = None
        selected_ids = []
        for i in range(self.args['open_slt_step']):
            rest_ids = list(np.setdiff1d(np.arange(txt_feats.shape[0]), np.array(selected_ids)))
            if all_feats is None:
                cand_ids = random.sample(rest_ids, int(self.args['open_slt_batch'] * self.args['open_slt_rate']))
                all_feats = txt_feats[np.array(cand_ids), :]
                continue
            cand_ids = random.sample(rest_ids, min(self.args['open_slt_batch'], len(rest_ids)))
            cand_batch = txt_feats[np.array(cand_ids), :]
            with torch.no_grad():
                all_feats = all_feats.to(self._device)
                cand_batch = cand_batch.to(self._device)
                cont_loss = cont_loss_fn(cand_batch, all_feats).detach().cpu().numpy()
            slt_ids = np.argsort(cont_loss)[:max(int(self.args['open_slt_batch'] * self.args['open_slt_rate']), 1)]
            all_feats = torch.cat([all_feats, cand_batch[slt_ids, :]], dim=0)
        rand_subset = np.array(random.sample(list(range(all_feats.shape[0])), feat_size))
        all_feats = all_feats[rand_subset, :]
        return all_feats


class LocalCELoss(torch.nn.Module):
    def __init__(self, start_class, end_class):
        super(LocalCELoss, self).__init__()
        self.start_class = start_class
        self.end_class = end_class
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        # x = x[:, self.start_class:]
        # y = y - self.start_class
        # change to mask, the same as original implementation
        x[:, :self.start_class] = float('-inf')
        loss = self.ce_loss(x, y.long())
        return loss


class SimpleReplayDataset(object):
    def __init__(self, mem_data, transforms):
        self.mem_sps, self.mem_labs = mem_data
        self.transforms = transforms
        self.mem_cnt = len(self.mem_sps)

    def get_batch(self, size):
        train_samples = []
        train_labels = []
        rand_ids = np.random.randint(low=0, high=self.mem_cnt, size=[size])
        for idx in rand_ids:
            sp = self.mem_sps[int(idx)]
            lab = self.mem_labs[int(idx)]
            if self.transforms is not None:
                train_samples.append(self.transforms(sp))
            else:
                train_samples.append(sp)
            train_labels.append(int(lab))
        train_samples = torch.stack(train_samples, dim=0)
        train_labels = torch.tensor(train_labels, dtype=torch.long, requires_grad=False)
        return train_samples, train_labels

    def get_data_by_class(self, ci):
        class_data = []
        for i in range(self.mem_cnt):
            lab = int(self.mem_labs[i])
            if lab == ci:
                if self.transforms is not None:
                    class_data.append(self.transforms(self.mem_sps[i]))
                else:
                    class_data.append(self.mem_sps[i])
        class_data = torch.stack(class_data, dim=0)
        return class_data


class FKDInputHook(object):
    def __init__(self, module, train=True):
        self.module = module
        self.inputs = None
        self.handle = self.module.register_forward_hook(hook=self.get_input_hook())
        self.train = train

    def get_feature(self):
        if self.train:
            return self.inputs
        else:
            return self.inputs.detach()

    def get_input_hook(self):

        def hook(module, input, output):
            self.inputs = input[0]

        return hook

    def remove_hook(self):
        self.inputs = None
        self.module = None
        self.handle.remove()


class RandomDataset(Dataset):
    def __init__(self, samples, labels, transforms):
        self.samples = samples
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.samples) * 10

    def __getitem__(self, index):
        idx = index % len(self.samples)
        if self.transforms is not None:
            sp = self.transforms(self.samples[idx])
        else:
            sp = self.samples[idx]
        lab = int(self.labels[idx])
        return sp, lab


def rotate_features(pro_a, pro_b, x):
    # assume a, b, x are unit (or treat them as embeddings on sphere)
    cos_t = np.dot(pro_a, pro_b)
    theta = np.arccos(cos_t)
    if np.isclose(theta, 0):
        return x.copy()
    v = pro_b - cos_t * pro_a
    v /= np.linalg.norm(v)
    # components in the {a, v} plane
    xa, xv = np.dot(pro_a, x), np.dot(v, x)
    # apply 2D rotation in that plane
    x_plane_rot = (cos_t * xa - np.sin(theta) * xv) * pro_a + (np.sin(theta) * xa + cos_t * xv) * v
    # remainder (orthogonal to both a, v) is unchanged
    x_ortho = x - xa * pro_a - xv * v
    return x_plane_rot + x_ortho
