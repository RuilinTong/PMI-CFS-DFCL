# -*-coding:utf8-*-
# functions for finetuning (R-DFCIL)

import torch
from optimization import lr_scheduler
from inversion import functions
import numpy as np


def _create_finetuning_lr_scheduler(optimizer: torch.optim.Optimizer, finetuning_epochs, max_epochs,
                                    finetuning_lr, base_lr):
    num_ft_epochs = finetuning_epochs
    if not (0 < num_ft_epochs < max_epochs):
        return None

    scheduler_ft = lr_scheduler.ConstantLR(
        optimizer,
        factor=finetuning_lr / base_lr,
        total_iters=num_ft_epochs,
        last_epoch=-2,  # prevent the step during initialization
    )

    return scheduler_ft


def add_finetuning_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                                finetuning_epochs, max_epochs, finetuning_lr, base_lr):
    scheduler_ft = _create_finetuning_lr_scheduler(
        optimizer,
        finetuning_epochs=finetuning_epochs,
        max_epochs=max_epochs,
        finetuning_lr=finetuning_lr,
        base_lr=base_lr
    )
    if scheduler_ft is None:
        return scheduler

    num_ft_epochs = finetuning_epochs
    milestones = [max_epochs - num_ft_epochs]
    scheduler = lr_scheduler.SequentialLR(
        optimizer, [scheduler, scheduler_ft], milestones
    )

    return scheduler


def finetuning(backbone, head, model_old, class_count, optimizer, scheduler, train_loader, epochs, loss_factor,
               hkd_factor, start_class, hkd_loss_fn, generator=None, use_cuda=True, loss_dic=None, train_params=None):
    print('\t === start finetuning ===')
    # freeze backbone
    # finetuning_state = {}
    # finetuning_state["backbone"] = functions.freeze(backbone)
    # TODO: add feature-kd when backbone training is included.
    for e in range(epochs):
        class_weights = class_count.sum() / class_count.clamp(min=1)
        class_weights = class_weights.div(class_weights.min())
        if use_cuda:
            class_weights = class_weights.cuda()
        step = 0
        for data in train_loader:
            sp, lab = data
            if use_cuda:
                sp = sp.cuda()
                lab = lab.cuda()
            feat = backbone(sp)
            out = head(feat)
            target_all = lab
            if generator is not None:
                inv_data = generator.get_data(batch_size=sp.shape[0])
                inv_sp, inv_lab, logit = inv_data
                feat_inv = backbone(inv_sp)
                out_inv = head(feat_inv)
                target_all = torch.cat([lab, inv_lab])
                out = torch.cat([out, out_inv], dim=0)
                ce_loss = torch.nn.functional.cross_entropy(out, target_all, class_weights)
                if (loss_dic is not None and 'hkd_loss' in loss_dic) or hkd_loss_fn is not None:
                    l_hkd = hkd_loss_fn(out_inv[:, :start_class], logit)
                    loss = loss_factor * ce_loss + hkd_factor * l_hkd
                    if step % 100 == 0:
                        print('finetuning loss at step', step, 'is:', loss.item(),
                              'ce loss:', ce_loss.item(), 'hkd loss:', l_hkd.item())
                else:
                    loss = loss_factor * ce_loss
                    if step % 100 == 0:
                        print('finetuning loss at step', step, 'is:', ce_loss.item())
            else:
                loss = loss_factor * torch.nn.functional.cross_entropy(out, target_all, class_weights)
                if step % 100 == 0:
                    print('finetuning loss at step', step, 'is:', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            indices, counts = target_all.cpu().unique(return_counts=True)
            class_count[indices] += counts
            step += 1
        scheduler.step()
    # unfreeze backbone
    # functions.unfreeze(backbone, finetuning_state.pop("backbone", {}))
    return backbone, head


def buffer_finetuning(backbone, head, model_old, class_count, optimizer, scheduler, train_loader, epochs, loss_factor,
                      hkd_factor, start_class, hkd_loss_fn, buffer, use_cuda=True):
    print('\t === start finetuning ===')
    # freeze backbone
    finetuning_state = {}
    finetuning_state["backbone"] = functions.freeze(backbone)
    for e in range(epochs):
        class_weights = class_count.sum() / class_count.clamp(min=1)
        class_weights = class_weights.div(class_weights.min())
        if use_cuda:
            class_weights = class_weights.cuda()
        step = 0
        for data in train_loader:
            sp, lab = data
            if use_cuda:
                sp = sp.cuda()
                lab = lab.cuda()
            feat = backbone(sp)
            out = head(feat)
            target_all = lab
            if buffer is not None:
                inv_sp, inv_lab = buffer.get_buffer_data(batch_size=sp.shape[0])
                if use_cuda:
                    inv_sp = inv_sp.cuda()
                    inv_lab = inv_lab.cuda()
                feat_inv = backbone(inv_sp)
                out_inv = head(feat_inv)
                target_all = torch.cat([lab, inv_lab])
                out = torch.cat([out, out_inv], dim=0)
                loss = loss_factor * torch.nn.functional.cross_entropy(out, target_all, class_weights)
                l_hkd = hkd_loss_fn(out_inv[:, :start_class], model_old(inv_sp))
                loss = loss + hkd_factor * l_hkd
            else:
                loss = loss_factor * torch.nn.functional.cross_entropy(out, target_all, class_weights)

            if step % 100 == 0:
                print('finetuning loss at step', step, 'is:', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            indices, counts = target_all.cpu().unique(return_counts=True)
            class_count[indices] += counts
            step += 1
        scheduler.step()
    # unfreeze backbone
    functions.unfreeze(backbone, finetuning_state.pop("backbone", {}))
    return backbone, head


def select_replay_samples(backbone, head, train_params, inputs, targets, slt_loss, ce_loss_fn, l1_loss_fn, logit=None):
    slt_num = int(inputs.shape[0] * train_params['slt_rate'])
    cur_out = head(backbone(inputs)).detach()
    cur_loss = ce_loss_fn(cur_out, targets).detach()
    if logit is not None:
        logit_dim = logit.shape[1]
        hkd_loss = torch.mean(l1_loss_fn(cur_out[:, :logit_dim], logit), dim=1)
    else:
        hkd_loss = 0
    ce_factor = train_params['ftslt_ce_factor']
    l1_factor = train_params['ftslt_l1_factor']
    loss_diff = ce_factor * (cur_loss - slt_loss) + l1_factor * hkd_loss
    loss_diff = loss_diff.cpu().numpy()
    slt_ids = np.flip(np.argsort(loss_diff), axis=0).copy()[:slt_num]
    slt_inputs = inputs[slt_ids, :]
    slt_targets = targets[slt_ids]
    return slt_inputs, slt_targets
