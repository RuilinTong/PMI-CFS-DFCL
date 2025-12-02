# -*-coding:utf8-*-

import os
import pickle
import torch
import math
import copy
from collections import OrderedDict
import numpy as np

from continual_runner import losses
from continual_runner import finetuning
from continual_runner import class_wise_contrastive_buffer
from continual_runner import cl_functions
import utils


class TaskClassContrastiveRunner(object):
    def __init__(self, local_path, train_params, dataset, inversion_params, task_dic, buffer_params,
                 cont_params, ce_mode='local', ckpt=None):
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.train_params = train_params
        self.inversion_params = inversion_params
        self.dataset = dataset
        self.task_dic = task_dic
        self.total_class = 0
        for tid in self.task_dic.keys():
            self.total_class += len(task_dic[tid])
        self.buffer_params = buffer_params
        self.cont_params = cont_params
        self.ce_mode = ce_mode
        self.ckpt = ckpt
        # build model
        if ckpt is not None:  # add support for loading checkpoint.
            backbone_file = os.path.join(self.local_path, 'backbone_' + str(ckpt) + '.pkl')
            head_file = os.path.join(self.local_path, 'head_' + str(ckpt) + '.pkl')
            self.backbone = torch.load(backbone_file)
            self.head = torch.load(head_file)
        else:
            self.backbone, self.head = utils.build_model(dataset=self.dataset, nf=self.train_params['nf'])
        old_model = [("backbone", self.backbone), ("head", self.head)]
        self.old_model = copy.deepcopy(torch.nn.Sequential(OrderedDict(old_model))).eval()
        if ckpt is not None:  # add support for loading buffer
            buffer_file = os.path.join(self.local_path, 'buffer_object_' + str(ckpt) + '.pkl')
            with open(buffer_file, 'rb') as fr:
                self.inv_buffer = pickle.load(fr)
            # update local path for gadi
            self.inv_buffer.local_path = os.path.join(self.local_path, 'buffer')
            self.inv_buffer.generator.local_path = os.path.join(self.local_path, 'buffer', 'generator')
        else:
            self.inv_buffer = class_wise_contrastive_buffer.ClassWiseContrastiveBuffer(
                local_path=os.path.join(self.local_path, 'buffer'),
                teacher_model=copy.deepcopy(self.old_model),
                dataset=self.dataset,
                inversion_params=self.inversion_params,
                buffer_params=self.buffer_params,
                eval_epoch=40,
                relabel_input=False,
                cont_params=cont_params
            )
        self.rkd_loss = torch.nn.Identity()
        self.feature_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.to_cuda()
        self.cur_layers = []
        self.feature_hooks = []
        self.teacher_hooks = []

    def to_cuda(self):
        if self.train_params['use_cuda']:
            self.backbone.cuda()
            self.head.cuda()
            self.old_model.cuda()

    def disable_backbone_training(self):
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def enable_backbone_training(self):
        self.backbone.train()
        self.backbone.requires_grad_(True)

    def train_head(self, optimizer, data_loader, test_loaders, epoch, loss_fn, ft_loss_fn, ce_factor, ft_factor,
                   hkd_loss_fn, hkd_factor, start_class, cls_count):
        for e in range(epoch):
            step = 0
            for data in data_loader:
                sp, lab = data
                if self.train_params['use_cuda']:
                    sp = sp.cuda()
                    lab = lab.cuda()
                feat = self.backbone(sp)
                out_logit = self.head(feat)
                l_ce = loss_fn(out_logit, lab)
                inv_data = self.inv_buffer.get_data(batch_size=sp.shape[0])
                inv_sp, inv_lab, teacher_logit = inv_data
                target_all = torch.cat([lab, inv_lab])
                feat_inv = self.backbone(inv_sp)
                inv_out = self.head(feat_inv)
                l_hkd = hkd_loss_fn(inv_out[:, :start_class], teacher_logit)
                # compute fine-tuning loss
                feat_detach = feat.detach()
                feat_inv_detach = feat_inv.detach()
                out_ft = self.head(torch.cat([feat_detach, feat_inv_detach], dim=0))
                lab_ft = target_all
                l_ft = ft_loss_fn(out_ft, lab_ft)
                loss = ce_factor * l_ce + hkd_factor * l_hkd * ft_factor * l_ft
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                indices, counts = target_all.cpu().unique(return_counts=True)
                cls_count[indices] += counts
                if step % 100 == 0:
                    print('CE loss at step', step, 'is:', l_ce.item(), 'HKD loss is:', l_hkd.item(),
                          'FT loss:', l_ft.item())
                step += 1
            if (e + 1) % 40 == 0 and test_loaders is not None:
                accs = self.evaluation(test_loaders=test_loaders, backbone=self.backbone, head=self.head)
                print('accuracies at epoch', e + 1, 'is:', accs)
                print('all:', np.mean(accs), 'new:', accs[-1], 'old:', np.mean(accs[:-1]))

    def train_with_backbone(self, optimizer, train_loader, test_loaders, epoch, ce_loss_fn, hkd_loss_fn, ft_loss_fn,
                            start_class, ce_factor, ft_factor, hkd_factor, rkd_factor, task_id, cls_count,
                            scheduler=None):
        for e in range(epoch):
            step = 0
            for data in train_loader:
                sp, lab = data
                if self.train_params['use_cuda']:
                    sp = sp.cuda()
                    lab = lab.cuda()
                feat = self.backbone(sp)
                out = self.head(feat)
                if task_id > 0:
                    inv_data = self.inv_buffer.get_data(batch_size=sp.shape[0])
                    inv_sp, inv_lab, teacher_logit = inv_data
                    target_all = torch.cat([lab, inv_lab])
                    # compute CE loss
                    feat_inv = self.backbone(inv_sp)
                    out_inv = self.head(feat_inv)
                    if self.ce_mode == 'local':
                        l_ce = ce_loss_fn(out, lab)
                        loss = ce_factor * l_ce
                    else:
                        l_ce = ce_loss_fn(torch.cat([out, out_inv], dim=0), target_all)
                        loss = ce_factor * l_ce
                    # compute hard knowledge distillation loss
                    l_hkd = hkd_loss_fn(out_inv[:, :start_class], teacher_logit)
                    # compute relation knowledge distillation loss
                    feat_new = self.feature_pool(feat)[:, :, 0, 0]
                    self.old_model.head.feature_mode = True
                    feat_old = self.old_model(sp).clone().detach()
                    self.old_model.head.feature_mode = False
                    l_rkd = self.rkd_loss(feat_new, feat_old)
                    # compute fine-tuning loss
                    feat_detach = feat.detach()
                    feat_inv_detach = feat_inv.detach()
                    out_ft = self.head(torch.cat([feat_detach, feat_inv_detach], dim=0))
                    lab_ft = target_all
                    l_ft = ft_loss_fn(out_ft, lab_ft)
                    loss = loss + hkd_factor * l_hkd + rkd_factor * l_rkd * ft_factor * l_ft
                    if step % 100 == 0:
                        print('ce loss:', l_ce.item(), 'hkd loss:', l_hkd.item(), 'rkd loss:', l_rkd.item(),
                              'ft loss:', l_ft.item())
                else:
                    l_ce = ce_loss_fn(out, lab)
                    loss = ce_factor * l_ce
                    target_all = lab
                if step % 100 == 0:
                    print('total loss at step', step, 'is:', loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                indices, counts = target_all.cpu().unique(return_counts=True)
                cls_count[indices] += counts
                step += 1
            if scheduler is not None:
                scheduler.step()
            # evaluation
            if (e + 1) % self.train_params['eval_epoch'] == 0 and test_loaders is not None:
                accs = self.evaluation(test_loaders=test_loaders, backbone=self.backbone, head=self.head)
                print('accuracies at epoch', e + 1, 'is:', accs)
                if task_id > 0:
                    print('Task:', task_id, 'all:', np.mean(accs), 'new:', accs[-1], 'old:', np.mean(accs[:-1]))
        return cls_count

    def train_single_task(self, train_loader, task_id, test_loaders):
        if task_id > 0:
            self.prepare_before_training(task_id=task_id)
            gpu_mem_info = utils.check_gpu_memory()
            print('gpu memory usage after inversion:', gpu_mem_info)
        self.add_class(task_id=task_id)
        start_class = 0
        for i in range(task_id):
            start_class += len(self.task_dic[i])
        end_class = start_class + len(self.task_dic[task_id])
        if self.ce_mode == 'local':
            ce_loss_fn = losses.LocalCELoss(start_class=start_class, end_class=end_class)
        else:
            ce_loss_fn = torch.nn.CrossEntropyLoss()
        hkd_loss_fn = torch.nn.L1Loss()

        if 'ft_beta' in self.train_params and self.train_params['ft_beta'] < 1.0:
            # add support for class balance
            ft_loss_fn = ClassBalanceCELoss(
                reduction='mean', use_cuda=self.train_params['use_cuda'],
                beta=self.train_params['ft_beta'], num_classes=self.head.num_classes)
        else:
            ft_loss_fn = torch.nn.CrossEntropyLoss()
        # make loss factors
        alpha = math.log(len(self.task_dic[task_id]) / 2 + 1, 2)
        beta = math.sqrt(start_class / len(self.task_dic[task_id]))
        if start_class == 0:
            ce_factor = 1
            hkd_factor = 0
            rkd_factor = 0
            ft_factor = 0
        else:
            if 'max_ratio' in self.train_params and self.train_params['max_ratio'] > 0:
                ce_factor = self.train_params['lambda_ce']
                ft_factor = self.train_params['lambda_ft']
                hkd_factor = self.train_params['lambda_hkd'] * \
                    min(self.train_params['max_ratio'], 1.0 + task_id * self.train_params['ratio_inc'])
                rkd_factor = self.train_params['lambda_rkd'] * \
                    min(self.train_params['max_ratio'], 1 + task_id * self.train_params['ratio_inc'])
            else:
                ce_factor = self.train_params['lambda_ce'] * (1 + 1 / alpha) / beta
                ft_factor = self.train_params['lambda_ft'] * (1 + 1 / alpha) / beta
                hkd_factor = self.train_params['lambda_hkd'] * alpha * beta
                rkd_factor = self.train_params['lambda_rkd'] * alpha * beta
        print('loss factor is:', ce_factor, hkd_factor, rkd_factor)
        # finetuning related class weights
        cls_count = torch.ones(self.head.num_classes)
        if task_id > 0:
            self.disable_backbone_training()
            next_stage = ''
            while next_stage is not None:
                # epoch iteration
                if len(self.cur_layers) == 0:
                    optimizer, scheduler = self.build_optimizer(
                        lr=self.train_params['lrs'][len(self.cur_layers)],
                        epochs=self.train_params['epochs'][0], use_rkd=False
                    )
                    self.train_head(
                        optimizer, data_loader=train_loader, test_loaders=None,
                        epoch=self.train_params['epochs'][0], loss_fn=ce_loss_fn, ft_loss_fn=ft_loss_fn,
                        ce_factor=ce_factor, ft_factor=ft_factor, hkd_factor=hkd_factor * 0.01, hkd_loss_fn=hkd_loss_fn,
                        start_class=start_class, cls_count=cls_count
                    )
                else:
                    optimizer, scheduler = self.build_optimizer(
                        lr=self.train_params['lrs'][len(self.cur_layers)],
                        epochs=self.train_params['epochs'][len(self.cur_layers)], use_rkd=True
                    )
                    cls_count = self.train_with_backbone(
                        optimizer=optimizer, train_loader=train_loader, test_loaders=test_loaders,
                        epoch=self.train_params['epochs'][len(self.cur_layers)], ce_loss_fn=ce_loss_fn,
                        ft_loss_fn=ft_loss_fn, hkd_loss_fn=hkd_loss_fn, start_class=start_class, ce_factor=ce_factor,
                        ft_factor=ft_factor, hkd_factor=hkd_factor, rkd_factor=rkd_factor, task_id=task_id,
                        cls_count=cls_count, scheduler=scheduler
                    )
                del optimizer
                next_stage = self.get_next_stage()
                self.cur_layers.append(next_stage)
            self.cur_layers.clear()
            self.enable_backbone_training()
        else:
            module = torch.nn.ModuleList([self.backbone, self.head, self.rkd_loss])
            optimizer = torch.optim.SGD(
                module.parameters(),
                lr=self.train_params['base_lr'],
                momentum=self.train_params['momentum'],
                weight_decay=self.train_params['weight_decay'],
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.train_params['milestones'],
                gamma=self.train_params['lr_factor'],
            )
            self.train_with_backbone(
                optimizer=optimizer, train_loader=train_loader, test_loaders=test_loaders,
                epoch=self.train_params['max_epochs'], ce_loss_fn=ce_loss_fn, ft_loss_fn=ft_loss_fn,
                hkd_loss_fn=hkd_loss_fn, start_class=start_class, ce_factor=ce_factor, ft_factor=ft_factor,
                hkd_factor=hkd_factor, rkd_factor=rkd_factor, task_id=task_id, cls_count=cls_count, scheduler=scheduler
            )
        print('class count after training is:', cls_count.numpy())
        # add procedure for finetuning.
        if self.train_params['finetuning_epochs'] > 0 and task_id > 0:
            self.disable_backbone_training()
            optimizer = torch.optim.SGD(
                self.head.parameters(),
                lr=self.train_params['finetuning_lr'],
                momentum=self.train_params['momentum'],
                weight_decay=self.train_params['weight_decay'],
            )
            if 'tune_stages' in self.train_params and self.train_params['tune_stages'] > 0:
                for j in range(self.train_params['tune_stages']):
                    stage = self.get_next_stage()
                    self.cur_layers.append(stage)
                # add parameters
                for si in self.cur_layers:
                    stage = getattr(self.backbone, si)
                    stage.train()
                    stage.requires_grad_(True)
                    optimizer.add_param_group(param_group={'params': stage.parameters()})
                    if si == 'stage_1':
                        print('add all parameters of model')
                        optimizer.add_param_group(param_group={'params': self.backbone.conv_1_3x3.parameters()})
                        optimizer.add_param_group(param_group={'params': self.backbone.bn_1.parameters()})
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=self.train_params['finetuning_epochs'],
                last_epoch=-1,
                verbose=False
            )
            # scheduler.step(0)
            self.backbone, self.head = finetuning.finetuning(
                backbone=self.backbone,
                head=self.head,
                model_old=self.old_model,
                class_count=cls_count,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                epochs=self.train_params['finetuning_epochs'],
                loss_factor=self.train_params['lambda_ce'],
                hkd_factor=hkd_factor,
                start_class=start_class,
                hkd_loss_fn=hkd_loss_fn,
                generator=self.inv_buffer,
                use_cuda=self.train_params['use_cuda'],
                train_params=None
            )
            print('class count after finetuning is:', cls_count.numpy())
            self.cur_layers.clear()
            self.enable_backbone_training()
        accs = self.evaluation(test_loaders=test_loaders, backbone=self.backbone, head=self.head)
        print('accuracies after training is:', accs)
        if task_id > 0:
            print('Task:', task_id, 'all:', np.mean(accs), 'new:', accs[-1], 'old:', np.mean(accs[:-1]))
        if 'save_data' in self.buffer_params and self.buffer_params['save_data']:
            self.inv_buffer.dump_data(task_id=task_id)
        self.update_buffer_params(train_loader=train_loader, start_class=start_class, end_class=end_class)
        return accs

    def train_single_task_no_gradual(self, train_loader, task_id, test_loaders):
        if task_id > 0:
            self.prepare_before_training(task_id=task_id)
        self.add_class(task_id=task_id)
        start_class = 0
        for i in range(task_id):
            start_class += len(self.task_dic[i])
        end_class = start_class + len(self.task_dic[task_id])
        if self.ce_mode == 'local':
            ce_loss_fn = losses.LocalCELoss(start_class=start_class, end_class=end_class)
        else:
            ce_loss_fn = torch.nn.CrossEntropyLoss()
        hkd_loss_fn = torch.nn.L1Loss()
        ft_loss_fn = torch.nn.CrossEntropyLoss()
        # test function: add feature level KD
        fkd_loss_fn = losses.FeatureRelationKDLoss()
        # make loss factors
        alpha = math.log(len(self.task_dic[task_id]) / 2 + 1, 2)
        beta = math.sqrt(start_class / len(self.task_dic[task_id]))
        if start_class == 0:
            ce_factor = 1
            hkd_factor = 0
            rkd_factor = 0
            ft_factor = 0
            fkd_factor = 0
        else:
            ce_factor = self.train_params['lambda_ce'] * (1 + 1 / alpha) / beta
            ft_factor = self.train_params['lambda_ft'] * (1 + 1 / alpha) / beta
            hkd_factor = self.train_params['lambda_hkd'] * alpha * beta
            rkd_factor = self.train_params['lambda_rkd'] * alpha * beta
            fkd_factor = self.train_params['lambda_fkd']
        print('loss factor is:', ce_factor, hkd_factor, rkd_factor)
        # finetuning related class weights
        cls_count = torch.ones(self.head.num_classes)
        if task_id > 0:
            while True:
                next_stage = self.get_next_stage()
                if next_stage is None:
                    break
                self.cur_layers.append(next_stage)
            optimizer, scheduler = self.build_optimizer(
                lr=self.train_params['lrs'][1], epochs=self.train_params['epochs'][1],
                use_rkd=True, milestones=self.train_params['milestones']
            )
            cls_count = self.train_with_backbone(
                optimizer=optimizer, train_loader=train_loader, test_loaders=test_loaders,
                epoch=self.train_params['epochs'][1], ce_loss_fn=ce_loss_fn,
                ft_loss_fn=ft_loss_fn, hkd_loss_fn=hkd_loss_fn, start_class=start_class, ce_factor=ce_factor,
                ft_factor=ft_factor, hkd_factor=hkd_factor, rkd_factor=rkd_factor, task_id=task_id,
                cls_count=cls_count, scheduler=scheduler
            )
            self.cur_layers.clear()
        else:
            module = torch.nn.ModuleList([self.backbone, self.head, self.rkd_loss])
            self.disable_backbone_training()
            optimizer = torch.optim.SGD(
                module.parameters(),
                lr=self.train_params['base_lr'],
                momentum=self.train_params['momentum'],
                weight_decay=self.train_params['weight_decay'],
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.train_params['milestones'],
                gamma=self.train_params['lr_factor'],
            )
            self.train_with_backbone(
                optimizer=optimizer, train_loader=train_loader, test_loaders=test_loaders,
                epoch=self.train_params['max_epochs'], ce_loss_fn=ce_loss_fn, ft_loss_fn=ft_loss_fn,
                hkd_loss_fn=hkd_loss_fn, start_class=start_class, ce_factor=ce_factor, ft_factor=ft_factor,
                hkd_factor=hkd_factor, rkd_factor=rkd_factor, task_id=task_id, cls_count=cls_count, scheduler=scheduler
            )
        print('class count after training is:', cls_count.numpy())
        # add procedure for finetuning.
        if self.train_params['finetuning_epochs'] > 0 and task_id > 0:
            # add support for finetuning on stage 4.
            optimizer = torch.optim.SGD(
                self.head.parameters(),
                lr=self.train_params['finetuning_lr'],
                momentum=self.train_params['momentum'],
                weight_decay=self.train_params['weight_decay'],
            )
            if 'tune_stages' in self.train_params and self.train_params['tune_stages'] > 0:
                for j in range(self.train_params['tune_stages']):
                    stage = self.get_next_stage()
                    self.cur_layers.append(stage)
                # add parameters
                for si in self.cur_layers:
                    stage = getattr(self.backbone, si)
                    stage.train()
                    stage.requires_grad_(True)
                    optimizer.add_param_group(param_group={'params': stage.parameters()})
                    if si == 'stage_1':
                        print('add all parameters of model')
                        optimizer.add_param_group(param_group={'params': self.backbone.conv_1_3x3.parameters()})
                        optimizer.add_param_group(param_group={'params': self.backbone.bn_1.parameters()})
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=self.train_params['finetuning_epochs'],
                last_epoch=-1,
                verbose=False
            )
            # scheduler.step(0)
            self.backbone, self.head = finetuning.finetuning(
                backbone=self.backbone,
                head=self.head,
                model_old=self.old_model,
                class_count=cls_count,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                epochs=self.train_params['finetuning_epochs'],
                loss_factor=self.train_params['lambda_ce'],
                hkd_factor=hkd_factor,
                start_class=start_class,
                hkd_loss_fn=hkd_loss_fn,
                generator=self.inv_buffer,
                use_cuda=self.train_params['use_cuda'],
                train_params=None
            )
            print('class count after finetuning is:', cls_count.numpy())
            self.cur_layers.clear()
            self.enable_backbone_training()
        accs = self.evaluation(test_loaders=test_loaders, backbone=self.backbone, head=self.head)
        print('accuracies after training is:', accs)
        if task_id > 0:
            print('Task:', task_id, 'all:', np.mean(accs), 'new:', accs[-1], 'old:', np.mean(accs[:-1]))
        if 'save_data' in self.buffer_params and self.buffer_params['save_data']:
            self.inv_buffer.dump_data(task_id=task_id)
        self.update_buffer_params(train_loader=train_loader, start_class=start_class, end_class=end_class)
        return accs

    def update_buffer_params(self, train_loader, start_class, end_class):
        """
        after training current task, and after training the first task
        :return:
        """
        old_model = [("backbone", self.backbone), ("head", self.head)]
        self.old_model = copy.deepcopy(torch.nn.Sequential(OrderedDict(old_model))).eval()
        if self.train_params['use_cuda']:
            self.old_model.cuda()
        # compute class mean and std for current task.
        cur_cls2mean, cur_cls2std = cl_functions.get_class_wise_distribution(
            backbone=self.old_model.backbone, data_loader=train_loader,
            classes=list(range(start_class, end_class)), on_cuda=self.train_params['use_cuda']
        )
        if start_class > 0:
            # compute class mean and std for previous data
            buffer_loader = self.inv_buffer.make_data_loader()
            prv_cls2mean, prv_cls2std = cl_functions.get_class_wise_distribution(
                backbone=self.old_model.backbone, data_loader=buffer_loader,
                classes=list(range(start_class)), on_cuda=self.train_params['use_cuda']
            )
        else:
            prv_cls2mean = {}
            prv_cls2std = {}
        all_cls2mean = {}
        all_cls2std = {}
        for ci in prv_cls2mean.keys():
            all_cls2mean[ci] = prv_cls2mean[ci]
            all_cls2std[ci] = prv_cls2std[ci]
        for ci in cur_cls2mean.keys():
            all_cls2mean[ci] = cur_cls2mean[ci]
            all_cls2std[ci] = cur_cls2std[ci]
        self.inv_buffer.update_buffer(teacher_model=self.old_model, train_loader=train_loader)

    def prepare_before_training(self, task_id):
        self.inv_buffer.generate_data_by_selection(
            selection_agent=None,
            samples=self.buffer_params['init_samples'] + task_id * self.buffer_params['per_task_samples']
        )
        self.rkd_loss = losses.RelationKDLoss(
            input_dim=self.backbone.num_features, output_dim=2 * self.backbone.num_features)
        if self.train_params['use_cuda']:
            self.rkd_loss.cuda()

    def get_next_stage(self):
        all_attributes = dir(self.backbone)
        stages = []
        for ati in all_attributes:
            if 'stage' in ati and ati not in self.cur_layers:
                stages.append(ati)
        if len(stages) == 0:
            return None
        sorted_stages = sorted(stages, reverse=True)
        next_stage = sorted_stages[0]
        return next_stage

    def build_optimizer(self, lr, epochs, use_rkd=True, milestones=None):
        if use_rkd:
            module = torch.nn.ModuleList([self.head.classifiers[-1], self.rkd_loss])
        else:
            module = self.head.classifiers[-1]
        optimizer = torch.optim.SGD(
            module.parameters(),
            lr=lr,
            momentum=self.train_params['momentum'],
            weight_decay=self.train_params['weight_decay'],
        )
        for si in self.cur_layers:
            stage = getattr(self.backbone, si)
            stage.train()
            stage.requires_grad_(True)
            optimizer.add_param_group(param_group={'params': stage.parameters()})
            if si == 'stage_1':
                print('add all parameters of model')
                optimizer.add_param_group(param_group={'params': self.backbone.conv_1_3x3.parameters()})
                optimizer.add_param_group(param_group={'params': self.backbone.bn_1.parameters()})
        total_params = 0
        for group in optimizer.param_groups:
            total_params += len(group['params'])
        print('Number of trainable parameters:', total_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[int(epochs * 0.5), int(epochs * 0.75)] if milestones is None else milestones,
            gamma=0.1
        )
        return optimizer, scheduler

    def evaluation(self, backbone, head, test_loaders, return_loss=False):
        backbone.eval()
        head.eval()
        accs = []
        task_loss = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():
            for ti in test_loaders:
                correct = 0
                total = 0
                total_loss = 0
                for data in ti:
                    sp, lab = data
                    if self.train_params['use_cuda']:
                        sp = sp.cuda()
                        lab = lab.cuda()
                    out = head(backbone(sp)).detach()
                    loss = loss_fn(out, lab).detach()
                    if self.train_params['use_cuda']:
                        lab = lab.cpu()
                    out = out.clone().detach().cpu()
                    loss = loss.clone().detach().cpu()
                    pred = out.argmax(dim=1, keepdim=True)
                    correct += pred.eq(lab.view_as(pred)).sum().item()
                    total_loss += loss.item()
                    total += sp.shape[0]
                acc = 100.0 * correct / total
                avg_loss = total_loss / total
                accs.append(acc)
                task_loss.append(avg_loss)
        backbone.train()
        head.train()
        if return_loss:
            return accs, task_loss
        return accs

    def add_class(self, task_id):
        self.head.append(num_classes=len(self.task_dic[task_id]))
        if self.train_params['use_cuda']:
            self.head.cuda()

    def compute_stat_loss(self, loss_fn):
        kd_loss = 0
        for i in range(len(self.feature_hooks)):
            loss = loss_fn(self.feature_hooks[i].get_feature(), self.teacher_hooks[i].get_feature())
            kd_loss = kd_loss + loss
        return kd_loss

    def end_training(self):
        if self.train_params['use_cuda']:
            self.backbone.cpu()
            self.head.cpu()
            self.rkd_loss.cpu()

    def dump_model(self, task_id):
        backbone_dump_file = os.path.join(self.local_path, 'backbone_' + str(task_id) + '.pkl')
        save_backbone = copy.deepcopy(self.backbone).cpu().eval()
        torch.save(save_backbone, backbone_dump_file)
        del save_backbone
        head_dump_file = os.path.join(self.local_path, 'head_' + str(task_id) + '.pkl')
        save_head = copy.deepcopy(self.head).cpu().eval()
        torch.save(save_head, head_dump_file)

    def save_buffer(self, task_id):
        print('--save buffer ---')
        save_file = os.path.join(self.local_path, 'buffer_object_' + str(task_id) + '.pkl')
        self.inv_buffer.prepare_for_save()  # remove inversion data
        with open(save_file, 'wb') as fw:
            pickle.dump(self.inv_buffer, fw)


class ClassBalanceCELoss(torch.nn.Module):
    def __init__(self, reduction, use_cuda, beta, num_classes):
        super().__init__()
        self.reduction = reduction
        self.use_cuda = use_cuda
        self.beta = beta  # smoothness factor, more smooth with greater beta
        self.cls_count = torch.ones(num_classes)
        if use_cuda:
            self.cls_count = self.cls_count.cuda()

    def forward(self, x, y):
        class_weights = self.cls_count.sum() / self.cls_count.clamp(min=1)
        class_weights = class_weights.div(class_weights.max())
        class_weights = 1.0 * self.beta + (1.0 - self.beta) * class_weights
        class_weights = class_weights.detach()
        loss = torch.nn.functional.cross_entropy(x, y, class_weights, reduction=self.reduction)
        # update class count
        indices, counts = y.unique(return_counts=True)
        self.cls_count[indices] += counts
        return loss
