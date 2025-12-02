# -*-coding:utf8-*-

import logging
import pickle
import numpy as np
import torch
import json
import time
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
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
from inversion import layer_wise_clip_inversion
from inversion import feature_stats
from inversion import cont_model
from inversion import utils
from inversion import building_blocks
from inversion import interfence_selection


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def ema_update(model, ema_model, ema_decay):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = ema_decay * ema_param.data + (1.0 - ema_decay) * param.data


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

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(train_loader, self.test_loader, tb_logger)
        if self._memory_size > 0 and self._total_classes < data_manager.n_classes:
            # change to inversion
            if bool(self.args['data_stat']):
                # update input feature stats according to real data
                self.inversion_runner.update_input_stat(data_loader=train_loader)
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
            if self.args['feat_type'] == 'cont':
                self.train_cont_models(
                    cls2feats=cls2feats, start_class=self._known_classes, end_class=self._total_classes)
            if self.args['int_tune_step'] <= 0:
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
        if 'save_ckpt' in self.args and bool(self.args['save_ckpt']):  # add support for saving checkpoint
            print('--- save checkpoint ---')
            save_path = os.path.join(self.local_path, 'model_' + str(self._cur_task) + '.pkl')
            save_model = copy.deepcopy(self._network).eval().cpu()
            torch.save(save_model, save_path)
            stat_path = os.path.join(self.local_path, 'stats_' + str(self._cur_task) + '.pkl')
            with open(stat_path, 'wb') as fw:
                pickle.dump(self.cls2stat, fw)
            # save block input stats
            layer_stats_folder = os.path.join(self.local_path, 'layer_stats_' + str(self._cur_task) + '.pkl')
            if not os.path.exists(layer_stats_folder):
                os.makedirs(layer_stats_folder)
            buffer_path = os.path.join(self.local_path, 'buffer')
            for fi in os.listdir(buffer_path):
                if 'input_stats' in fi and fi.endswith('.pkl'):
                    src_file = os.path.join(buffer_path, fi)
                    dst_file = os.path.join(layer_stats_folder, fi)
                    shutil.copy(src_file, dst_file)

    def _train(self, train_loader, test_loader, tb_logger=None):
        self._network.to(self._device)
        # add function of pseudo tuning
        if self.args['int_tune_step'] > 0 and self._known_classes > 0:
            if self.args['vis_tune_blocks'] > 0 or self.args['txt_tune_blocks'] > 0:
                tuned_blocks, tuned_ft, vis_layer_names, ori_blocks, ori_ft = self.pseudo_finetuning(
                    train_loader=train_loader)
            else:
                tuned_blocks, tuned_ft, vis_layer_names, ori_blocks, ori_ft = self.interference_no_tuning()
            # build selection agent
            slt_agent = interfence_selection.InterferenceSelectionAgent(
                tuned_blocks=tuned_blocks, ori_blocks=ori_blocks, layer_names=vis_layer_names,
                select_rate=self.args['int_slt_rate'], device=self._device, ori_text_feat=ori_ft, text_feat=tuned_ft,
                know_class=self._total_classes, logit_scale=self._network.logit_scale.exp(),
                batch_dim=1 if self.args['vis_tune_blocks'] > 0 else 0, head=None, slt_mode=self.args['int_slt_mode']
            )  # tuning blocks should be less than 11
            mem_sample, mem_label = self.build_inv_data(
                cls2stat=self.cls2stat, int_slt_params={'selection_agent': slt_agent})
            self._data_memory = mem_sample
            self._targets_memory = mem_label
            # update buffer
            self.buffer = SimpleReplayDataset(
                mem_data=[self._data_memory, self._targets_memory],
                transforms=train_loader.dataset.trsf if self.args['start_block'] == 0 else None
            )
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
        # add layer-wise feature KD loss
        layer_fkd_loss_fn = torch.nn.MSELoss()
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
            layer_kd_factor = self.args['lambda_lkd']
        else:
            ce_factor = 1.0
            hkd_factor = self.args['lambda_hkd']
            layer_kd_factor = self.args['lambda_lkd']
            tkd_factor = self.args['lambda_tkd']
        if layer_kd_factor > 0:
            self.register_hooks()
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
                    if layer_kd_factor > 0:
                        loss_lkd = self.compute_stat_loss(loss_fn=layer_fkd_loss_fn)
                        loss = loss + layer_kd_factor * loss_lkd
                    else:
                        loss_lkd = torch.tensor(0, dtype=torch.float32, requires_grad=False)
                    if iteration % 100 == 0:
                        print('losses at step', iteration, 'hkd loss:', loss_hkd.item(),
                              'lkd loss:', loss_lkd.item(), 'tkd loss:', loss_tkd.item())

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
        if layer_kd_factor > 0:
            self.remove_hooks()
        self._network.eval()
        self._network.set_is_train(is_train=None)
        if self._known_classes > 0 and self.args['finetune_epoch'] > 0:
            self.finetune_classifier(train_loader=train_loader, class_count=cls_count)
        # check the effect of memory data
        if self.args['mem_tune_step'] > 0 and self._known_classes > 0:
            self.tune_on_memory()

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

    def register_hooks(self):
        """
        for feature level knowledge distillation
        :return:
        """
        teacher_model = self.inv_model
        for n, mi in teacher_model.named_modules():
            if 'transformer.resblocks.' in n and len(n.split('.')) == 3:
                self.teacher_hooks.append(FKDInputHook(module=mi, train=False))
        for n, mi in self._network.visual.named_modules():
            if 'transformer.resblocks.' in n and len(n.split('.')) == 3:
                self.feature_hooks.append(FKDInputHook(module=mi, train=True))

    def remove_hooks(self):
        for hi in self.feature_hooks:
            hi.remove_hook()
        self.feature_hooks.clear()
        for hi in self.teacher_hooks:
            hi.remove_hook()
        self.teacher_hooks.clear()

    def compute_stat_loss(self, loss_fn):
        kd_loss = 0
        for i in range(len(self.feature_hooks)):
            loss = loss_fn(self.feature_hooks[i].get_feature(), self.teacher_hooks[i].get_feature())
            kd_loss = kd_loss + loss
        return kd_loss

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

    def build_inv_data(self, cls2stat, int_slt_params=None):
        print('number of inversion class:', len(cls2stat))
        train_samples = []
        train_labels = []
        target_feats = []
        for ci in cls2stat.keys():
            # random sample text features
            stat = cls2stat[ci]
            feat_size = self.args['memory_per_class']
            if self.args['int_tune_step'] > 0:
                feat_size = int(feat_size // self.args['int_slt_rate'])
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
        # select feat for inversion
        if self.args['int_tune_step'] > 0 and int_slt_params is not None:
            print('--- select interfered samples ---')
            all_loss_diffs = []
            if self.args['vis_tune_blocks'] > 0:  # add support for only tuning text encoder
                sp_cnt = 0
                while sp_cnt < target_feats.shape[0]:
                    bs = min(self.args['gen_batch_size'] * 2, target_feats.shape[0] - sp_cnt)
                    layer_batch = min(self.args['layer_batch'] * 2, target_feats.shape[0] - sp_cnt)
                    slt_ids, loss_diffs = self.inversion_runner.partial_inversion_and_select(
                        batch_size=bs,
                        target_feats=target_feats[sp_cnt:sp_cnt + layer_batch],
                        targets=torch.tensor(train_labels[sp_cnt:sp_cnt + layer_batch],
                                             dtype=torch.long, requires_grad=False),
                        finetune_iters=self.args['tune_steps'],
                        finetune_lr=self.args['tune_lr'],
                        milestones=[int(mi) for mi in self.args['milestones'].split(',')],
                        selection_agent=int_slt_params['selection_agent'],
                        iters=self.args['train_steps'],
                        return_best=True,
                        verbose=False
                    )
                    all_loss_diffs.append(loss_diffs)
                    sp_cnt += layer_batch
                all_loss_diffs = np.concatenate(all_loss_diffs, axis=0)
                slt_ids = interfence_selection.class_balance_selection(
                    loss_diffs=all_loss_diffs, targets=train_labels, know_class=self._known_classes,
                    slt_num=self.args['memory_per_class'] * self._known_classes
                )
            else:
                _, all_loss_diffs = int_slt_params['selection_agent'].select_by_interference(
                    in_feats=target_feats, targets=torch.tensor(train_labels, dtype=torch.long, requires_grad=False))
                slt_ids = interfence_selection.class_balance_selection(
                    loss_diffs=all_loss_diffs, targets=train_labels, know_class=self._known_classes,
                    slt_num=self.args['memory_per_class'] * self._known_classes
                )
            train_labels = train_labels[slt_ids]
            target_feats = target_feats[slt_ids, :]
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

    def train_cont_models(self, cls2feats, start_class, end_class):
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
            self.cont_models.append(trained_model)
            os.remove(temp_file)

    def pseudo_finetuning(self, train_loader):
        print('--- pseudo finetuning ---')
        # tune last few layers for interference selection
        ce_factor = self.args['lambda_ce']
        self._network.set_is_train(is_train=None)
        temp_model = copy.deepcopy(self._network)
        cnn_accy, nme_accy = self.eval_task(network=temp_model)
        print('accuracies before pseudo tuning:', cnn_accy["grouped"])
        # build optimizer only tune last few layers
        visual_res_blocks = len(temp_model.visual.transformer.resblocks)
        text_res_blocks = len(temp_model.transformer.resblocks)
        if self.args['vis_tune_blocks'] > 0:  # add support for only tune text encoder
            tune_vis_ids = list(range(visual_res_blocks))[-self.args['vis_tune_blocks']:]
        else:
            tune_vis_ids = []
        if self.args['txt_tune_blocks'] > 0:
            tune_txt_ids = list(range(text_res_blocks))[-min(self.args['txt_tune_blocks'], text_res_blocks):]
        else:
            tune_txt_ids = []
        vis_layer_names = []
        for li in tune_vis_ids:
            vis_layer_names.append('resblocks.' + str(li))
        txt_layer_names = []
        for li in tune_txt_ids:
            txt_layer_names.append('resblocks.' + str(li))
        params = []
        for k, v in temp_model.named_parameters():
            if k.startswith('visual'):
                tune_layer = False
                for li in vis_layer_names:
                    if li in k:
                        tune_layer = True
                        break
                if ("adaptmlp" in k or "router" in k or "noise" in k) and tune_layer:
                    params.append(v)
            else:
                tune_layer = False
                for li in txt_layer_names:
                    if li in k:
                        tune_layer = True
                        break
                if ("adaptmlp" in k or "router" in k or "noise" in k) and tune_layer:
                    params.append(v)
        print('parameters:', len(params))
        optimizer = torch.optim.AdamW(params, lr=self.args['int_tune_lr'], weight_decay=self.args['weight_decay'])
        num_batches = len(train_loader)
        total_iterations = self.args['tuned_epoch'] * num_batches
        scheduler = cosine_lr(optimizer, self.args['init_lr'], 30, total_iterations)
        steps = 0
        cur_texts = self.tokenizer(self.cur_texts).cuda().to(self._device)
        for e in range(self.args['tuned_epoch']):
            for data in train_loader:  # maximum one epoch
                scheduler(steps % len(train_loader))
                _, inputs, targets = data
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, _ = temp_model(inputs, cur_texts, 0, is_train=True)
                optimizer.zero_grad()
                loss_ce = F.cross_entropy(
                    logits, targets.long() - self._known_classes, label_smoothing=self.args['label_smoothing'])
                loss = ce_factor * loss_ce
                loss.backward()
                optimizer.step()
                steps += 1
                if steps == self.args['int_tune_step']:
                    break
        temp_model.eval()
        temp_model.set_is_train(is_train=False)
        cnn_accy, nme_accy = self.eval_task(network=temp_model)
        print('accuracies after pseudo tuning:', cnn_accy["grouped"])
        # return image encoder, text feature of all existing classes
        all_texts = self.tokenizer(self.all_texts).cuda().to(self._device)
        tuned_ft = temp_model.encode_text(all_texts, normalize=True, is_train=False).detach()
        ori_ft = self.old_model.encode_text(all_texts, normalize=True, is_train=False).detach()
        tuned_blocks = []
        ori_blocks = []
        if self.args['vis_tune_blocks'] > 0:
            for n, mi in temp_model.visual.named_modules():
                if 'transformer.resblocks.' in n and len(n.split('.')) == 3:
                    for li in vis_layer_names:
                        if li in n:
                            tuned_blocks.append(copy.deepcopy(mi))
                            break
            # post process
            post_trans_block = building_blocks.PostTransformBlock(
                proj=copy.deepcopy(temp_model.visual.proj), ln_post=copy.deepcopy(temp_model.visual.ln_post),
                normalize=True
            )
            tuned_blocks.append(post_trans_block)
            for n, mi in self.inv_model.named_modules():
                if 'transformer.resblocks.' in n and len(n.split('.')) == 3:
                    for li in vis_layer_names:
                        if li in n:
                            ori_blocks.append(mi)
                            break
            ori_post_trans_block = building_blocks.PostTransformBlock(
                proj=copy.deepcopy(self.inv_model.proj), ln_post=copy.deepcopy(self.inv_model.ln_post),
                normalize=True
            )
            ori_blocks.append(ori_post_trans_block)
        del temp_model
        return tuned_blocks, tuned_ft, vis_layer_names, ori_blocks, ori_ft

    def interference_no_tuning(self):
        all_texts = self.tokenizer(self.all_texts).cuda().to(self._device)
        prv_texts = self.tokenizer(self.all_texts[:self._known_classes]).cuda().to(self._device)
        new_ft = self._network.encode_text(all_texts, normalize=True, is_train=False).detach()
        prv_ft = self._network.encode_text(prv_texts, normalize=True, is_train=False).detach()
        cmp_ft = torch.zeros([self._total_classes - self._known_classes, new_ft.shape[1]], dtype=torch.float32)
        cmp_ft = cmp_ft.to(self._device)
        prv_ft = torch.cat([prv_ft, cmp_ft], dim=0)
        return [], new_ft, [], [], prv_ft

    def tune_on_memory(self):
        # only tune model on memory data
        print('--- memory finetuning ---')
        hkd_factor = self.args['lambda_hkd']
        tkd_factor = self.args['lambda_tkd']
        self._network.set_is_train(is_train=None)
        if bool(self.args['tune_on_model']):
            temp_model = self._network
        else:
            temp_model = copy.deepcopy(self._network)
        cnn_accy, nme_accy = self.eval_task(network=temp_model)
        temp_model.train()
        temp_model.set_is_train(is_train=True)
        print('accuracies before memory tuning:', cnn_accy["grouped"])
        params = [v for k, v in temp_model.named_parameters()
                  if ("adaptmlp" in k or "router" in k or "noise" in k) and k.startswith('visual')]
        print('parameters:', len(params))
        optimizer = torch.optim.AdamW(params, lr=self.args['init_lr'], weight_decay=self.args['weight_decay'])
        total_iterations = self.args['mem_tune_step']
        scheduler = cosine_lr(optimizer, self.args['init_lr'], 30, total_iterations)
        # add hard KD loss
        if self.args['hkd_loss'] == 'l1':
            hkd_loss_fn = torch.nn.L1Loss()
        else:
            hkd_loss_fn = torch.nn.MSELoss()
        if self.args['tkd_loss'] == 'l1':
            tkd_loss_fn = torch.nn.L1Loss()
        else:
            tkd_loss_fn = torch.nn.MSELoss()
        if self.args['start_block'] > 0 and self._known_classes > 0:  # only used for replay
            cur_blocks, batch_dim = utils.split_clip_blocks(
                model=temp_model.visual, split_cnn=True, normalize=True)
            prv_blocks, batch_dim = utils.split_clip_blocks(model=self.inv_model, split_cnn=True, normalize=True)
            cur_model = torch.nn.Sequential(*cur_blocks[self.args['start_block']:])
            prv_model = torch.nn.Sequential(*prv_blocks[self.args['start_block']:])
        else:
            cur_model = None
            prv_model = None
        old_texts = self.tokenizer(self.all_texts[:self._known_classes]).cuda().to(self._device)
        for i in range(self.args['mem_tune_step']):
            scheduler(i % 78)
            # hkd loss
            logit_scale = temp_model.logit_scale.exp()
            old_ft = self.old_model.encode_text(old_texts, normalize=True, is_train=False).detach()
            mem_sp, mem_lab = self.buffer.get_batch(size=self.args['batch_size'])
            mem_sp, mem_lab = mem_sp.to(self._device), mem_lab.to(self._device)
            if self.args['start_block'] == 0:  # consider MoE case
                old_fi = temp_model.encode_image(mem_sp, normalize=True, is_train=True)
                old_logit = logit_scale * old_fi @ old_ft.t()
                teacher_fi, _ = self.inv_model(mem_sp).detach()
                teacher_fi = teacher_fi / teacher_fi.norm(dim=-1, keepdim=True)
            else:
                old_fi = cur_model(mem_sp)  # add support for is_train
                teacher_fi = prv_model(mem_sp).detach()
                old_logit = logit_scale * old_fi @ old_ft.t()
            teacher_logit = logit_scale * teacher_fi @ old_ft.t()
            loss_hkd = hkd_factor * hkd_loss_fn(old_logit, teacher_logit)
            loss_hkd.backward()
            # add text replay.
            # student_ft = temp_model.encode_text(old_texts, normalize=True, is_train=True)
            # teacher_ft = self.old_model.encode_text(old_texts, normalize=True, is_train=False).detach()
            # loss_tkd = tkd_factor * tkd_loss_fn(student_ft, teacher_ft)
            # loss_tkd.backward()
            if i % 100 == 0 or i == self.args['mem_tune_step'] - 1:
                print('losses at step', i, 'hkd loss:', loss_hkd.item())
            optimizer.step()
        temp_model.set_is_train(is_train=None)
        temp_model.eval()
        cnn_accy, nme_accy = self.eval_task(network=temp_model)
        print('accuracies after memory tuning:', cnn_accy["grouped"])
        if not bool(self.args['tune_on_model']):
            del temp_model


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
