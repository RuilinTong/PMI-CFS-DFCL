# -*-coding:utf8-*-
# Layer-wise CLIP inversion without using generator

import os
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchvision
import copy
import pickle
import numpy as np

from inversion import functions
from inversion import input_stats
from inversion import utils


class LayerWiseCLIPInversion(object):
    """
    model: visual encoder of CLIP model.
    """
    def __init__(self, local_path, model, image_size, lr, train_steps, alpha_pr, alpha_rf, scheduler_params,
                 use_rf=True, smooth_type='tv', flip_rate=0, log_step=1000, opt_type='adam', boost_factor=True,
                 loss_type='l2', grad_norm=None, clip_input='clip', input_aug=False, save_step=0,
                 pre_size_change=None, normalize=True, head=None, rf_factor=1.0, start_block=0):
        self.local_path = local_path
        if self.local_path is not None and not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.input_resolution = 224
        self.model = model
        self.image_size = image_size
        self.lr = lr
        self.train_steps = train_steps
        self.alpha_pr = alpha_pr
        self.alpha_rf = alpha_rf
        self.cls2mean = []
        self.cls2std = []
        self.scheduler_params = scheduler_params
        self.use_rf = use_rf
        self.smooth_type = smooth_type
        self.flip_rate = flip_rate
        self.log_step = log_step
        self.opt_type = opt_type
        self.boost_factor = boost_factor
        self.loss_type = loss_type
        self.grad_norm = grad_norm
        self.clip_input = clip_input
        self.input_aug = input_aug
        self.save_step = save_step
        self.normalize = normalize
        self.head = head
        self.rf_factor = rf_factor
        self.start_block = start_block
        # hooks
        self.on_cuda = False
        self.stat_hooks = []
        self.gaussian_kernel = functions.Gaussiansmoothing(channels=3, kernel_size=5, sigma=1)
        if self.loss_type == 'l2':
            self.loss_fn = CLIPLoss()
        elif self.loss_type == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.loss_fn = CosineLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.pool_layer = torch.nn.AdaptiveAvgPool2d(1)
        self.best_layer_lrs = []
        self.best_layer_rfs = []
        self.tv_function = None
        if self.smooth_type == 'tv':
            self.tv_function = functions.TotalVariation()
        if self.input_aug and self.start_block == 0:  # consider start block
            self.in_aug = functions.CLIPInputAugmentation(
                size=self.input_resolution, flip_rate=self.flip_rate, size_change=pre_size_change)
        else:
            self.in_aug = functions.SimpleInputAugmentation(
                size=self.input_resolution if self.start_block == 0 else int(self.input_resolution // 16),
                size_change=pre_size_change,
                normalize=True if self.start_block == 0 else False
            )

    def get_input_stat(self):
        # prepare input stats
        input_stats.get_transformer_block_stats(
            model=self.model, samples=10000, img_size=[3, self.input_resolution, self.input_resolution],
            out_path=self.local_path, on_cuda=self.on_cuda
        )
        input_stats.get_block_input_stats(
            model=self.model, samples=10000, img_size=[3, self.input_resolution, self.input_resolution],
            out_path=self.local_path, on_cuda=self.on_cuda, split_cnn=False if self.start_block == 0 else True
        )

    def update_input_stat(self, data_loader):
        print('\t update input feature stats')
        ori_model_file = os.path.join(self.local_path, 'model_input_stats.pkl')
        ori_block_files = []
        for fi in os.listdir(self.local_path):
            if fi.startswith('block_input_stats') and fi.endswith('.pkl'):
                ori_block_files.append(os.path.join(self.local_path, fi))
        if not os.path.exists(ori_model_file) and len(ori_block_files) == 0:
            input_stats.get_data_transformer_block_stats(
                model=self.model, data_loader=data_loader, out_path=self.local_path,
                on_cuda=self.on_cuda, fname='model_input_stats.pkl'
            )
            input_stats.get_data_block_input_stats(
                model=self.model, data_loader=data_loader, out_path=self.local_path,
                on_cuda=self.on_cuda, fname='block_input_stats', split_cnn=False if self.start_block == 0 else True
            )
        else:
            input_stats.get_data_transformer_block_stats(
                model=self.model, data_loader=data_loader, out_path=self.local_path,
                on_cuda=self.on_cuda, fname='new_model_input_stats.pkl'
            )
            input_stats.get_data_block_input_stats(
                model=self.model, data_loader=data_loader, out_path=self.local_path,
                on_cuda=self.on_cuda, fname='new_block_input_stats', split_cnn=False if self.start_block == 0 else True
            )
            # merge stats
            input_stats.merge_stats(
                ori_stat_file=ori_model_file, new_stat_file=os.path.join(self.local_path, 'new_model_input_stats.pkl'))
            for i in range(14):
                ori_file = os.path.join(self.local_path, 'block_input_stats' + str(i) + '.pkl')
                if not os.path.exists(ori_file):
                    continue
                input_stats.merge_stats(
                    ori_stat_file=os.path.join(self.local_path, 'block_input_stats' + str(i) + '.pkl'),
                    new_stat_file=os.path.join(self.local_path, 'new_block_input_stats' + str(i) + '.pkl')
                )
                os.remove(os.path.join(self.local_path, 'new_block_input_stats' + str(i) + '.pkl'))

    def update_model(self, model, head=None):
        if self.on_cuda:
            self.model.cpu()
        self.model = model
        self.head = head
        if self.on_cuda:
            self.model.cuda()
            if self.head is not None:
                self.head.cuda()
        # need to tune parameters with model is updated
        self.best_layer_lrs.clear()
        self.best_layer_rfs.clear()

    def register_hooks(self):
        stat_file = os.path.join(self.local_path, 'model_input_stats.pkl')
        if not os.path.exists(stat_file):
            raise ValueError('Input stat file does not exist')
        with open(stat_file, 'rb') as fr:
            name2stat = pickle.load(fr)
        for n, mi in self.model.named_modules():
            if n in name2stat:
                self.stat_hooks.append(
                    functions.CustomBNInputHook(
                        module=mi,
                        running_mean=name2stat[n][0].cuda() if self.on_cuda else name2stat[n][0].cuda(),
                        running_var=name2stat[n][1].cuda() if self.on_cuda else name2stat[n][1].cuda(),
                        batch_dim=1  # only transformer blocks
                    )
                )

    def register_block_hooks(self, block, bid, batch_dim):
        stat_file = os.path.join(self.local_path, 'block_input_stats' + str(bid) + '.pkl')
        if not os.path.exists(stat_file):
            return
        with open(stat_file, 'rb') as fr:
            name2stat = pickle.load(fr)
        for n, mi in block.named_modules():
            if n in name2stat:
                self.stat_hooks.append(
                    functions.CustomBNInputHook(
                        module=mi,
                        running_mean=name2stat[n][0].cuda() if self.on_cuda else name2stat[n][0].cuda(),
                        running_var=name2stat[n][1].cuda() if self.on_cuda else name2stat[n][1].cuda(),
                        batch_dim=batch_dim  # only transformer blocks
                    )
                )

    def remove_hooks(self):
        for hi in self.stat_hooks:
            hi.remove_hook()
        self.stat_hooks.clear()

    def criterion_pr(self, inputs):
        if self.smooth_type == 'tv':
            pr_loss = self.tv_function(inputs)
        else:
            input_pad = F.pad(inputs, (2, 2, 2, 2), mode="reflect")
            input_smooth = self.gaussian_kernel(input_pad).detach()
            pr_loss = F.mse_loss(inputs, input_smooth)
        return pr_loss

    def inversion(self, inputs, target_feats, opt_param, loss_fn, model, size_change=None, is_input=False,
                  iters=None, norm_feat=True, return_best=False, input_loss=None, alpha_feat=None, l_pr=None,
                  in_aug=None, verbose=True, id_bias=None, full_model=True):
        best_loss = None
        best_inputs = None
        if iters is None:
            iters = self.train_steps
        opt, scheduler = self.build_optimizer(
            inputs=inputs,
            layer_wise=opt_param['layer_wise'],
            is_input=is_input,
            lr=opt_param['lr'],
            milestones=opt_param['milestones']
        )
        if alpha_feat is None:
            alpha_rf = self.alpha_rf
            alpha_in = self.alpha_rf
        else:
            alpha_rf = alpha_feat
            alpha_in = alpha_feat
        if l_pr is None:
            alpha_pr = self.alpha_pr
        else:
            alpha_pr = l_pr
        if in_aug is None:
            in_aug = self.in_aug
        alpha_mse = 1.0
        init_losses = {}
        for i in range(iters):
            if size_change is not None and i in size_change:
                if self.start_block == 0:  # consider start block
                    new_res = min(inputs.shape[2] * 2, 224)
                else:
                    new_res = min(inputs.shape[2] * 2, 14)
                print('\tup scale image to size:', new_res)
                up_sample = functions.Scale(new_res)
                inputs = up_sample(inputs.detach())
                inputs.requires_grad_(True)
                opt, scheduler = self.build_optimizer(
                    inputs=inputs,
                    layer_wise=opt_param['layer_wise'],
                    is_input=is_input,
                    lr=opt_param['lr'],
                    milestones=opt_param['milestones']
                )
            opt.zero_grad()
            # input augmentation
            if is_input:
                aug_inputs = in_aug(inputs)
            else:
                aug_inputs = inputs
            if full_model:
                out_feat = model(aug_inputs)[0]
            else:
                out_feat = model(aug_inputs)
            if norm_feat:
                normed_feat = out_feat / (out_feat.norm(dim=-1, keepdim=True) + 1e-6)
            else:
                normed_feat = out_feat
            if len(self.stat_hooks) > 0:
                l_stat = torch.stack([h.stats_regularization() for h in self.stat_hooks]).sum()
            else:
                l_stat = torch.tensor(0, dtype=torch.float32, requires_grad=False)
            if is_input:
                l_blur = self.criterion_pr(inputs=inputs)
            else:
                l_blur = torch.tensor(0, dtype=torch.float32, requires_grad=False)
            l_mse = loss_fn(normed_feat, target_feats)
            if i == 0:
                init_losses['mse'] = l_mse.item()
                init_losses['stat'] = l_stat.item()
            else:  # prevent non-convergence
                if l_mse.item() > init_losses['mse'] and self.boost_factor:
                    alpha_mse = min(alpha_mse * 2, 10)
                if l_stat.item() > init_losses['stat'] and self.boost_factor:
                    alpha_rf = min(alpha_rf * 2, 10 * self.alpha_rf)
            loss = alpha_mse * l_mse + alpha_rf * l_stat + alpha_pr * l_blur
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
            loss.backward()
            # gradient clip and image clip
            if is_input and self.grad_norm is not None:
                clip_grad_norm_([inputs], self.grad_norm)
            if is_input and self.start_block == 0:
                inputs.data = regularize_input(inputs=inputs, clip_type=self.clip_input)
            opt.step()
            if (i % self.log_step == 0 or i == iters - 1) and verbose:
                print('finish training step:', i)
                print('mse loss:', l_mse.item(), 'smooth loss:', l_blur.item(), 'stat loss:', l_stat.item(),
                      'total loss:', loss.item(), end=' ' if l_in is not None else '\n')
                if l_in is not None:
                    print('input loss:', l_in.item())
            if id_bias is not None and self.save_step > 0 and i % self.save_step == 0:
                save_images(
                    img_batch=inputs.data, out_path=self.local_path, step=i, id_bias=id_bias)
            cmp_loss = l_mse.item() + self.alpha_rf * l_stat.item() + self.alpha_pr * l_blur.item()
            if l_in is not None:
                cmp_loss += self.alpha_rf * l_in.item()
            if best_loss is None or best_loss > cmp_loss:
                best_loss = cmp_loss
                best_inputs = copy.deepcopy(inputs.clone().detach())
            if scheduler is not None:  # fix the problem of scheduler called before optimizer
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR) and i >= 3400:
                    scheduler.step()
                else:
                    scheduler.step()
        if return_best:
            return best_inputs, best_loss
        else:
            return inputs, best_loss

    def get_data(self, target_feats, size_change=None, iters=None, return_best=False, init_input=None):
        ori_start_block = self.start_block
        self.start_block = 0
        self.register_hooks()
        if self.on_cuda:
            target_feats = target_feats.cuda()
        shape = [target_feats.shape[0], ] + self.image_size
        if self.on_cuda:
            if init_input is None:
                inputs = torch.rand(shape, device='cuda', dtype=torch.float32).requires_grad_(True)
            else:  # option for initializing inputs
                inputs = torch.tensor(init_input, device='cuda', dtype=torch.float32, requires_grad=True)
        else:
            if init_input is None:
                inputs = torch.rand(shape, device='cpu', dtype=torch.float32).requires_grad_(True)
            else:  # option for initializing inputs
                inputs = torch.tensor(init_input, device='cpu', dtype=torch.float32, requires_grad=True)
        opt_param = {
            'layer_wise': False,
            'is_input': True,
            'lr': self.lr,
            'milestones': None
        }
        ori_alpha_rf = self.alpha_rf
        if not self.use_rf:
            self.alpha_rf = 0
        inputs, best_loss = self.inversion(
            inputs=inputs, target_feats=target_feats, opt_param=opt_param, size_change=size_change,
            norm_feat=self.normalize, is_input=True, return_best=return_best, iters=iters, loss_fn=self.loss_fn,
            model=self.model, id_bias=0
        )
        self.alpha_rf = ori_alpha_rf
        self.remove_hooks()
        self.in_aug.reset()
        inputs.clone().detach()
        self.start_block = ori_start_block
        return inputs

    def layer_wise_inversion_for_cl(self, batch_size, target_feats,  finetune_iters, finetune_lr, milestones,
                                    iters=None, size_change=None, return_best=True, first_layer_param=None,
                                    search_param=True, verbose=False, gradual_rf=False):
        """
        layer-wise inversion, do model inversion per layer, aims to accelerate model inversion.
        :param batch_size:
        :param target_feats:
        :param finetune_iters:
        :param finetune_lr:
        :param milestones:
        :param iters:
        :param size_change:
        :param return_best:
        :param search_param:
        :param first_layer_param:
        :param gradual_rf:
        :param verbose:
        :return:
        """
        if self.on_cuda:
            target_feats = target_feats.cuda()
        layer_batch = target_feats.shape[0]
        # get all blocks
        all_blocks, batch_dims = utils.split_clip_blocks(
            model=self.model, normalize=self.normalize, split_cnn=False if self.start_block == 0 else True)
        print('total blocks:', len(all_blocks))
        out_feats = target_feats
        # --- layer wise inversion ---
        shape = [target_feats.shape[0], ] + [3, self.input_resolution, self.input_resolution]
        out_shape = target_feats.shape
        opt_inputs = None
        # for different loss factor in layer-wise inversion
        self.alpha_rf = self.alpha_rf * len(all_blocks) * self.rf_factor
        ori_alpha_pr = self.alpha_pr
        self.alpha_pr = 0
        candidate_lrs = [self.lr * 2, self.lr, self.lr * 0.5, self.lr * 0.25, self.lr * 0.125]
        candidate_rfs = [self.alpha_rf, self.alpha_rf * 0.5, self.alpha_rf * 0.25, self.alpha_rf * 0.125]
        if not search_param:
            self.best_layer_lrs = [self.lr] * len(all_blocks)
            if gradual_rf:
                self.best_layer_rfs = list(np.flip(np.arange(len(all_blocks)) * self.alpha_rf / len(all_blocks)))
            else:
                self.best_layer_rfs = [self.alpha_rf] * len(all_blocks)
        for i in range(len(all_blocks) - 1, self.start_block - 1, -1):
            # if verbose:
            #     print('Inverting for block:', i)
            # get input and output shape
            if i > 0 and i == self.start_block:
                _, input_shape = self.get_module_input_shape(
                    module=all_blocks[i - 1], all_blocks=all_blocks[:self.start_block],
                    shape=[target_feats.shape[0], ] + self.image_size, pool_layer=None
                )
            else:
                input_shape, expect_out_shape = self.get_module_input_shape(
                    module=all_blocks[i], all_blocks=all_blocks, shape=shape, pool_layer=None)
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
                        device = 'cuda'
                    else:
                        device = 'cpu'
                    if i == 0:
                        inputs = torch.rand(
                            [target_feats.shape[0], ] + self.image_size, device=device,
                            dtype=torch.float32, requires_grad=True
                        )
                        input_loss = None
                    else:
                        inputs, input_loss = self.init_layer_input(
                            idx=i, input_shape=input_shape, batch_dim=batch_dims[i])
                    # build optimizer
                    i_iters, i_size_change, i_return_best, i_l_pr, is_input, i_in_aug, inv_lr = \
                        self.get_block_inv_settings(
                            bid=i, first_layer_param=first_layer_param, iters=iters, return_best=return_best, lr=lr)
                    opt_param = {
                        'layer_wise': True,
                        'is_input': is_input,
                        'lr': inv_lr,
                        'milestones': None
                    }
                    self.register_block_hooks(block=all_blocks[i], bid=i, batch_dim=batch_dims[i])
                    # model inversion training
                    opt_inputs, best_loss = self.inversion(
                        inputs=inputs,
                        target_feats=out_feats,
                        opt_param=opt_param,
                        model=all_blocks[i],
                        norm_feat=self.normalize if i == len(all_blocks) - 1 else False,
                        iters=i_iters,
                        size_change=i_size_change[0],
                        return_best=i_return_best,
                        input_loss=input_loss,
                        alpha_feat=alpha_rf,
                        l_pr=i_l_pr,
                        verbose=False,
                        loss_fn=self.loss_fn if i == len(all_blocks) - 1 else self.mse_loss,
                        is_input=is_input,
                        in_aug=i_in_aug,
                        full_model=False
                    )
                    self.remove_hooks()
                    best_loss_cross = 0
                    if best_cand_loss is None or best_cand_loss > (best_loss + best_loss_cross):
                        best_cand_loss = (best_loss + best_loss_cross)
                        best_lr = lr
                        best_alpha_rf = alpha_rf
                        best_inputs = opt_inputs.clone().detach()  # fix error of best inputs
            if len(self.best_layer_lrs) < len(all_blocks) or len(self.best_layer_rfs) < len(all_blocks):
                print('best learning rate and alpha rf for block', i, 'is:', best_lr, best_alpha_rf)
                self.best_layer_lrs = [best_lr] + self.best_layer_lrs
                self.best_layer_rfs = [best_alpha_rf] + self.best_layer_rfs
            opt_inputs = best_inputs
            # update output
            out_feats = opt_inputs.clone().detach()
            out_shape = opt_inputs.shape
        self.alpha_rf = self.alpha_rf / (len(all_blocks)) / (self.rf_factor + 1e-6)
        self.alpha_pr = ori_alpha_pr
        # finetune inputs with the full model
        print('model inversion on full model')
        init_input = opt_inputs.cpu().numpy()
        tuned_inputs = []
        steps = int(layer_batch // batch_size)
        if layer_batch % batch_size != 0:
            steps += 1
        ori_alpha_rf = self.alpha_rf
        if not self.use_rf:
            self.alpha_rf = 0
        for i in range(steps):
            start = i * batch_size
            end = min((i + 1) * batch_size, layer_batch)
            inputs = torch.tensor(init_input[start:end, :], device='cuda', dtype=torch.float32, requires_grad=True)
            opt_param = {
                'layer_wise': False,
                'is_input': True,
                'lr': finetune_lr,
                'milestones': milestones
            }
            self.register_hooks()
            # add choice of downscale image and gradual increase size
            opt_inputs, best_loss = self.inversion(
                inputs=inputs, target_feats=target_feats[start:end, :],
                opt_param=opt_param, size_change=size_change,
                norm_feat=self.normalize, is_input=True,
                return_best=return_best, iters=finetune_iters,
                model=self.model if self.start_block == 0 else torch.nn.Sequential(*all_blocks[self.start_block:]),
                full_model=True if self.start_block == 0 else False,
                verbose=verbose, loss_fn=self.loss_fn, id_bias=start, input_loss=None
            )
            self.remove_hooks()
            self.in_aug.reset()
            print('best loss after tuning is:', best_loss)
            opt_inputs.clone().cpu().detach()
            tuned_inputs.append(opt_inputs)
        self.alpha_rf = ori_alpha_rf
        tuned_inputs = torch.cat(tuned_inputs, dim=0)
        return tuned_inputs

    def partial_inversion_and_select(self, batch_size, target_feats, targets, finetune_iters, finetune_lr, milestones,
                                     selection_agent, iters=None, return_best=True, verbose=False):
        # add support for interference selection.
        if self.on_cuda:
            target_feats = target_feats.cuda()
        layer_batch = target_feats.shape[0]
        # get all blocks
        all_blocks, batch_dims = utils.split_clip_blocks(
            model=self.model, normalize=self.normalize, split_cnn=False if self.start_block == 0 else True)
        print('total blocks:', selection_agent.num_blocks)
        out_feats = target_feats
        # --- layer wise inversion ---
        shape = [target_feats.shape[0], ] + [3, self.input_resolution, self.input_resolution]
        opt_inputs = None
        # for different loss factor in layer-wise inversion
        self.alpha_rf = self.alpha_rf * len(all_blocks) * self.rf_factor
        ori_alpha_pr = self.alpha_pr
        self.alpha_pr = 0
        # per layer inversion
        for i in range(len(all_blocks) - 1, len(all_blocks) - selection_agent.num_blocks - 1, -1):
            if i > 0 and i == self.start_block:
                _, input_shape = self.get_module_input_shape(
                    module=all_blocks[i - 1], all_blocks=all_blocks[:self.start_block],
                    shape=[target_feats.shape[0], ] + self.image_size, pool_layer=None
                )
            else:
                input_shape, expect_out_shape = self.get_module_input_shape(
                    module=all_blocks[i], all_blocks=all_blocks, shape=shape, pool_layer=None)
            # initialize input
            if self.on_cuda:
                device = 'cuda'
            else:
                device = 'cpu'
            if i == 0:
                inputs = torch.rand(
                    [target_feats.shape[0], ] + self.image_size, device=device,
                    dtype=torch.float32, requires_grad=True
                )
                input_loss = None
            else:
                inputs, input_loss = self.init_layer_input(
                    idx=i, input_shape=input_shape, batch_dim=batch_dims[i])
            # build optimizer
            i_iters, i_size_change, i_return_best, i_l_pr, is_input, i_in_aug, inv_lr = \
                self.get_block_inv_settings(
                    bid=i, first_layer_param=None, iters=iters, return_best=return_best, lr=self.lr)
            opt_param = {
                'layer_wise': True,
                'is_input': is_input,
                'lr': inv_lr,
                'milestones': None
            }
            self.register_block_hooks(block=all_blocks[i], bid=i, batch_dim=batch_dims[i])
            # model inversion training
            opt_inputs, best_loss = self.inversion(
                inputs=inputs,
                target_feats=out_feats,
                opt_param=opt_param,
                model=all_blocks[i],
                norm_feat=self.normalize if i == len(all_blocks) - 1 else False,
                iters=i_iters,
                size_change=i_size_change[0],
                return_best=i_return_best,
                input_loss=input_loss,
                alpha_feat=self.alpha_rf,
                l_pr=i_l_pr,
                verbose=False,
                loss_fn=self.loss_fn if i == len(all_blocks) - 1 else self.mse_loss,
                is_input=is_input,
                in_aug=i_in_aug,
                full_model=False
            )
            self.remove_hooks()
            opt_inputs = opt_inputs
            # update output
            out_feats = opt_inputs.clone().detach()
        self.alpha_rf = self.alpha_rf / (len(all_blocks)) / (self.rf_factor + 1e-6)
        self.alpha_pr = ori_alpha_pr
        init_input = opt_inputs.cpu().numpy()
        tuned_inputs = []
        steps = int(layer_batch // batch_size)
        if layer_batch % batch_size != 0:
            steps += 1
        ori_alpha_rf = self.alpha_rf
        if not self.use_rf:
            self.alpha_rf = 0
        start_block = len(all_blocks) - selection_agent.num_blocks
        for i in range(steps):
            start = i * batch_size
            end = min((i + 1) * batch_size, layer_batch)
            if selection_agent.batch_dim == 0:
                inputs = torch.tensor(init_input[start:end, :], device='cuda', dtype=torch.float32, requires_grad=True)
            elif selection_agent.batch_dim == 1:
                inputs = torch.tensor(
                    init_input[:, start:end, :], device='cuda', dtype=torch.float32, requires_grad=True)
            else:
                raise ValueError('Invalid batch dim')
            opt_param = {
                'layer_wise': False,
                'is_input': True,
                'lr': finetune_lr,
                'milestones': milestones
            }
            # only register hooks of last few blocks.
            stat_file = os.path.join(self.local_path, 'model_input_stats.pkl')
            if not os.path.exists(stat_file):
                raise ValueError('Input stat file does not exist')
            with open(stat_file, 'rb') as fr:
                name2stat = pickle.load(fr)
            for n, mi in self.model.named_modules():
                tuned = False
                for li in selection_agent.layer_names:
                    if li in n:
                        tuned = True
                        break
                if n in name2stat and tuned:
                    self.stat_hooks.append(
                        functions.CustomBNInputHook(
                            module=mi,
                            running_mean=name2stat[n][0].cuda() if self.on_cuda else name2stat[n][0].cuda(),
                            running_var=name2stat[n][1].cuda() if self.on_cuda else name2stat[n][1].cuda(),
                            batch_dim=1  # only transformer blocks
                        )
                    )
            opt_inputs, best_loss = self.inversion(
                inputs=inputs, target_feats=target_feats[start:end, :],
                opt_param=opt_param, size_change=None,
                norm_feat=self.normalize, is_input=False,
                return_best=return_best, iters=finetune_iters,
                model=torch.nn.Sequential(*all_blocks[start_block:]),
                full_model=False, verbose=verbose, loss_fn=self.loss_fn, id_bias=start, input_loss=None
            )
            self.remove_hooks()
            self.in_aug.reset()
            print('best loss after tuning is:', best_loss)
            opt_inputs.clone().cpu().detach()
            tuned_inputs.append(opt_inputs)
        self.alpha_rf = ori_alpha_rf
        if selection_agent.batch_dim == 0:
            tuned_inputs = torch.cat(tuned_inputs, dim=0)
        elif selection_agent.batch_dim == 1:
            tuned_inputs = torch.cat(tuned_inputs, dim=1)
        else:
            raise ValueError('Invalid batch dim')
        # select feature by interference
        slt_ids, loss_diffs = selection_agent.select_by_interference(in_feats=tuned_inputs, targets=targets)
        return slt_ids, loss_diffs

    def build_optimizer(self, inputs, layer_wise, is_input, lr, milestones):
        if self.opt_type == 'adam':
            opt = torch.optim.Adam([inputs], lr=lr)
        else:
            opt = torch.optim.SGD([inputs], lr=lr)
        if layer_wise:
            if not is_input:
                scheduler = None
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=opt,
                    milestones=milestones,
                    gamma=0.5
                )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
        return opt, scheduler

    def get_block_inv_settings(self, bid, first_layer_param, iters, return_best, lr):
        if first_layer_param is not None and bid == self.start_block:
            first_iter = first_layer_param['iters']
            l_sc = first_layer_param['size_change']
            l_pr = first_layer_param['alpha_pr']
            if not first_layer_param['l_aug']:
                layer_aug = functions.SimpleInputAugmentation(
                    size=self.input_resolution if self.start_block == 0 else int(self.input_resolution // 16),
                    normalize=True if self.start_block == 0 else False)
                if self.on_cuda:
                    layer_aug.cuda()
            else:
                layer_aug = None
            l_lr = first_layer_param['l_lr']
        else:
            first_iter = iters
            l_sc = None
            l_pr = None
            layer_aug = None
            l_lr = self.lr
        iters = iters if bid > self.start_block else first_iter
        size_change = None if bid > self.start_block else l_sc,
        return_best = True if bid > self.start_block else return_best
        l_pr = None if bid > self.start_block else l_pr
        is_input = True if bid == self.start_block else False
        in_aug = layer_aug
        inv_lr = lr if bid > self.start_block else l_lr
        return iters, size_change, return_best, l_pr, is_input, in_aug, inv_lr

    def get_module_input_shape(self, all_blocks, module, shape, pool_layer=None):
        model = torch.nn.Sequential(*all_blocks)
        inputs = torch.rand(shape, dtype=torch.float32, requires_grad=False)
        if self.on_cuda:
            inputs = inputs.cuda()
        hook = InputShapeHook(module=module, pool_layer=pool_layer)
        with torch.no_grad():
            _ = model(inputs)
        input_shape, output_shape = hook.get_input_output_shape()
        hook.remove_hook()
        return input_shape, output_shape

    def init_layer_input(self, idx, input_shape, batch_dim):
        # The output of block is residual sum, better to compute the distribution of input
        stat_file = os.path.join(self.local_path, 'block_input_stats' + str(idx) + '.pkl')
        with open(stat_file, 'rb') as fr:
            name2stat = pickle.load(fr)
        stats = name2stat['total']
        bias, var = stats[0], stats[1]
        bias = bias.cpu().numpy()
        var = var.cpu().numpy()
        weight = np.sqrt(var)
        if batch_dim == 0:
            if len(input_shape) == 4:  # consider 4-dim input
                weight = np.expand_dims(weight, axis=[0, 2, 3])
                bias = np.expand_dims(bias, axis=[0, 2, 3])
            else:
                weight = np.expand_dims(weight, axis=[0, 2])
                bias = np.expand_dims(bias, axis=[0, 2])
            init_value = torch.randn(input_shape, device='cpu').numpy()
            init_input = init_value * weight + bias
        else:
            weight = np.expand_dims(weight, axis=[1, 2])
            bias = np.expand_dims(bias, axis=[1, 2])
            init_value = torch.randn(input_shape, device='cpu').numpy()
            init_input = init_value * weight + bias
        input_loss = KLDivergence(
            mean=torch.tensor(np.squeeze(bias), requires_grad=False, dtype=torch.float32),
            var=torch.tensor(np.squeeze(weight) ** 2, requires_grad=False, dtype=torch.float32),
            batch_dim=batch_dim
        )
        if self.on_cuda:
            inputs = torch.tensor(init_input, device='cuda', dtype=torch.float32, requires_grad=True)
            input_loss.cuda()
        else:
            inputs = torch.tensor(init_input, device='cpu', dtype=torch.float32, requires_grad=True)
            input_loss.cpu()
        return inputs, input_loss

    def get_sample_loss(self, samples, target_feats):
        loss_fn = CLIPLoss(reduction='none')
        steps = int(samples.shape[0] / 100)
        if samples.shape[0] % 100 != 0:
            steps += 1
        all_loss = []
        with torch.no_grad():
            for i in range(steps):
                start = 100 * i
                end = min(samples.shape[0], (i + 1) * 100)
                out_logit = self.model(samples[start:end, :]).detach()
                loss = loss_fn(out_logit, target_feats[start:end]).cpu().numpy()
                for j in range(loss.shape[0]):
                    all_loss.append(loss[j])
        return all_loss

    def cuda(self):
        self.model.cuda()
        if self.in_aug is not None:
            self.in_aug.cuda()
        self.on_cuda = True

    def cpu(self):
        self.model.cpu()
        if self.in_aug is not None:
            self.in_aug.cpu()
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
    def __init__(self, mean, var, batch_dim):
        self.mean = mean
        self.var = var
        self.batch_dim = batch_dim

    def compute_loss(self, x):
        # consider 4-dim input
        running_mean = self.mean
        running_var = self.var
        # get stats of input
        if len(x.shape) == 4:
            mean = x.mean([self.batch_dim, 2, 3])
        else:
            mean = x.mean([self.batch_dim, 2])
        if self.batch_dim == 0:
            nch = x.shape[1]
            if len(x.shape) == 4:
                value = x.permute(1, 0, 2, 3).contiguous().view([nch, -1])
            else:
                value = x.permute(1, 0, 2).contiguous().view([nch, -1])
        else:  # batch_dim == 1
            nch = x.shape[0]
            value = x.contiguous().view([nch, -1])
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


class CLIPLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.norm(x - y, dim=1, p=2)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class CosineLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = 1.0 - torch.sum(x * y, dim=1)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def regularize_input(inputs, clip_type):
    if clip_type == 'clip':
        inputs.data = torch.clip(inputs.data, 0, 1)
    elif clip_type == 'rescale':  # contain input pixels by re-scale.
        ch_min = torch.min(torch.min(inputs.data, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0].detach()
        ch_max = torch.max(torch.max(inputs.data, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0].detach()
        inputs.data = (inputs.data - ch_min) / (ch_max - ch_min)
    return inputs.data


def save_images(img_batch, out_path, step, id_bias):
    # img_batch.data = regularize_input(inputs=img_batch, clip_type=clip_type)
    for i in range(img_batch.shape[0]):
        save_file = os.path.join(out_path, str(i + id_bias) + '_' + str(step) + '.png')
        temp_img = copy.deepcopy(img_batch[i, :])
        ch_min = torch.min(torch.min(temp_img, dim=2, keepdim=True)[0], dim=1, keepdim=True)[0].detach()
        ch_max = torch.max(torch.max(temp_img, dim=2, keepdim=True)[0], dim=1, keepdim=True)[0].detach()
        temp_img = (temp_img - ch_min) / (ch_max - ch_min)
        torchvision.utils.save_image(temp_img, save_file, normalize=True, scale_each=True)
