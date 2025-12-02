# -*-coding:utf8-*-

import os
import argparse
import pickle

from continual_runner import task_contrastive_cl_runner
from datasets import seq_cifar100
from datasets import seq_tiny_imagenet
import utils


def main(opts):
    if not os.path.exists(opts.local_path):
        os.makedirs(opts.local_path)
    # make dataset
    with open(opts.class_order_file, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
        line = lines[0]
    class_order = [int(ci.strip()) for ci in line.split(', ')]
    if opts.dataset == 'seq_cifar100':
        generator = seq_cifar100.SeqCIFAR100(
            data_path=opts.data_path,
            download=True,
            class_order=class_order,
            tasks=opts.tasks,
            batch_size=opts.batch_size,
            workers=0,
            first_task_class=None if opts.init_split <= 0 else opts.init_split
        )
    elif opts.dataset == 'seq_tinyimagenet':
        generator = seq_tiny_imagenet.SeqTinyImageNet(
            data_path=opts.data_path,
            class_order=class_order,
            tasks=opts.tasks,
            batch_size=opts.batch_size,
            workers=0,
            first_task_class=None if opts.init_split <= 0 else opts.init_split
        )
    else:
        raise ValueError('Invalid data set')
    lrs = [float(lr) for lr in opts.lrs.split(',')]
    epochs = [int(ei) for ei in opts.epochs.split(',')]
    # make runner
    train_params = {
        'use_cuda': bool(opts.use_cuda),
        'base_lr': opts.base_lr,
        'lr_factor': opts.lr_factor,
        'lrs': lrs,
        'momentum': opts.momentum,
        'weight_decay': opts.weight_decay,
        'milestones': [int(mi) for mi in opts.milestones.split(',')],
        'finetuning_epochs': opts.finetuning_epochs,
        'finetuning_lr': opts.finetuning_lr,
        'tune_stages': opts.tune_stages,
        'max_epochs': opts.max_epochs,
        'epochs': epochs,
        'lambda_ce': opts.lambda_ce,
        'lambda_hkd': opts.lambda_hkd,
        'lambda_rkd': opts.lambda_rkd,
        'lambda_fkd': opts.lambda_fkd,
        'lambda_ft': opts.lambda_ft,
        'ft_beta': opts.ft_beta,
        'nf': opts.nf,
        'max_ratio': opts.max_ratio,
        'ratio_inc': opts.ratio_inc,
        'eval_epoch': opts.eval_epoch,
    }
    inversion_params = {
        'lr': opts.inversion_lr,
        'train_steps': opts.train_steps,
        'tune_steps': opts.tune_steps,
        'tune_lr': opts.tune_lr,
        'alpha_pr': opts.alpha_pr,
        'alpha_rf': opts.alpha_rf,
        'rf_factor': opts.rf_factor,
        'flip_rate': opts.flip_rate,
        'layer_wise': bool(opts.layer_wise),
        'layer_batch': opts.layer_batch,
        'boost_factor': bool(opts.boost_factor),
        'search_param': bool(opts.search_param),
        'scheduler_params': {
            'milestones': [int(ei) for ei in opts.inv_milestones.split(',')],
            'lr_rate': opts.inv_lr_rate,
            'warmup': opts.inv_warmup,
            'layer_schedule': bool(opts.layer_schedule)
        }
    }
    buffer_params = {
        'init_samples': opts.init_samples,
        'per_task_samples': opts.per_task_samples,
        'gen_batch_size': opts.gen_batch_size,
        'use_cuda': bool(opts.use_cuda),
        'save_data': bool(opts.save_data),
        'use_unslt': bool(opts.use_unslt),
        'feat_type': opts.feat_type
    }
    cont_params = {
        'slt_step': opts.cont_step,
        'slt_rate': opts.cont_rslt,
        'blocks': opts.cont_blocks,
        'lr': opts.cont_lr,
        'epoch': opts.cont_epoch,
        'tau': opts.cont_temp,
        'act': opts.act
    }
    runner = task_contrastive_cl_runner.TaskClassContrastiveRunner(
        local_path=opts.local_path,
        train_params=train_params,
        dataset=opts.dataset,
        buffer_params=buffer_params,
        inversion_params=inversion_params,
        task_dic=generator.get_task_dic(),
        ce_mode=opts.ce_mode,
        cont_params=cont_params,
        ckpt=opts.ckpt if len(opts.ckpt) > 0 else None
    )
    # continual training
    test_loaders = []
    task_accs = []
    start_task = 0
    if len(opts.ckpt) > 0:
        start_task = int(opts.ckpt) + 1
        for i in range(start_task):
            train_loader, test_loader = generator.get_task_loaders(task_id=i)
            test_loaders.append(test_loader)
    for i in range(start_task, opts.tasks):
        if i == opts.max_task:
            if bool(opts.save_ckpt):
                runner.save_buffer(task_id=i - 1)
            break
        train_loader, test_loader = generator.get_task_loaders(task_id=i)
        test_loaders.append(test_loader)
        if bool(opts.gradual):
            accs = runner.train_single_task(
                train_loader=train_loader,
                task_id=i,
                test_loaders=test_loaders
            )
        else:
            accs = runner.train_single_task_no_gradual(
                train_loader=train_loader,
                task_id=i,
                test_loaders=test_loaders
            )
        task_accs.append(accs)
        if bool(opts.save_ckpt):
            runner.dump_model(task_id=i)
    runner.end_training()
    dump_file = os.path.join(opts.local_path, 'accs.pkl')
    with open(dump_file, 'wb') as fw:
        pickle.dump(task_accs, fw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model inversion CL')
    # base parameter
    parser.add_argument('--local_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--use_cuda', type=int)
    parser.add_argument('--save_ckpt', type=int, default=0)
    parser.add_argument('--max_task', type=int, default=-1)
    parser.add_argument('--gradual', type=int, default=1)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--seed', type=int)
    # data parameter
    parser.add_argument('--init_split', type=int)
    parser.add_argument('--tasks', type=int)
    parser.add_argument('--class_order_file', type=str)
    parser.add_argument('--batch_size', type=int)
    # train parameter
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--lr_factor', type=float)
    parser.add_argument('--lrs', type=str)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--milestones', type=str)
    parser.add_argument('--finetuning_epochs', type=int)
    parser.add_argument('--finetuning_lr', type=float)
    parser.add_argument('--tune_stages', type=int, default=0)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--epochs', type=str)
    parser.add_argument('--lambda_ce', type=float)
    parser.add_argument('--lambda_hkd', type=float)
    parser.add_argument('--lambda_rkd', type=float)
    parser.add_argument('--lambda_fkd', type=float)
    parser.add_argument('--lambda_ft', type=float)
    parser.add_argument('--ft_beta', type=float, default=1.0)
    parser.add_argument('--ce_mode', type=str, default='local')
    parser.add_argument('--nf', type=int, default=16)
    parser.add_argument('--max_ratio', type=float, default=0.0)
    parser.add_argument('--ratio_inc', type=float, default=0.0)
    parser.add_argument('--eval_epoch', type=int, default=40)
    # buffer parameter
    parser.add_argument('--init_samples', type=int)
    parser.add_argument('--per_task_samples', type=int, default=0)
    parser.add_argument('--gen_batch_size', type=int)
    parser.add_argument('--save_data', type=int, default=0)
    parser.add_argument('--use_unslt', type=int, default=0)
    parser.add_argument('--feat_type', type=str, default='cont')
    # contrastive parameter
    parser.add_argument('--cont_lr', type=float, default=1e-2)
    parser.add_argument('--cont_epoch', type=int, default=200)
    parser.add_argument('--cont_blocks', type=int, default=2)
    parser.add_argument('--cont_step', type=int, default=100)
    parser.add_argument('--cont_rslt', type=float, default=0.5)
    parser.add_argument('--act', type=str, default='leaky')
    parser.add_argument('--cont_temp', type=float, default=1.0)
    # inversion parameter
    parser.add_argument('--inversion_lr', type=float)
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--tune_steps', type=int)
    parser.add_argument('--tune_lr', type=float)
    parser.add_argument('--alpha_pr', type=float)
    parser.add_argument('--alpha_rf', type=float)
    parser.add_argument('--rf_factor', type=float, default=1.0)
    parser.add_argument('--flip_rate', type=float, default=0)
    parser.add_argument('--layer_wise', type=int, default=1)
    parser.add_argument('--layer_batch', type=int)
    parser.add_argument('--boost_factor', type=int, default=1)
    parser.add_argument('--search_param', type=int, default=1)
    parser.add_argument('--inv_milestones', type=str)
    parser.add_argument('--inv_lr_rate', type=float, default=0.5)
    parser.add_argument('--inv_warmup', type=int, default=0)
    parser.add_argument('--layer_schedule', type=int, default=0)
    args = parser.parse_args()

    print('script\t\t', 'main_task_contrastive_cl.py')
    print('\n')
    print('--- base parameter ---')
    print('local path\t\t', args.local_path)
    print('dataset\t\t', args.dataset)
    print('data path\t\t', args.data_path)
    print('use cuda\t\t', args.use_cuda)
    print('save checkpoint\t\t', args.save_ckpt)
    print('max_task\t\t', args.max_task)
    print('gradual\t\t', args.gradual)
    print('ckpt\t\t', args.ckpt)
    print('seed\t\t', args.seed)
    print('\n')
    print('--- data parameter ---')
    print('init split\t\t', args.init_split)
    print('tasks\t\t', args.tasks)
    print('class order file\t\t', args.class_order_file)
    print('batch size\t\t', args.batch_size)
    print('\n')
    print('--- train parameter ---')
    print('base lr\t\t', args.base_lr)
    print('lr factor\t\t', args.lr_factor)
    print('lrs\t\t', args.lrs)
    print('momentum\t\t', args.momentum)
    print('weight decay\t\t', args.weight_decay)
    print('milestones\t\t', args.milestones)
    print('finetuning epochs\t\t', args.finetuning_epochs)
    print('finetuning lr\t\t', args.finetuning_lr)
    print('tune_stages\t\t', args.tune_stages)
    print('max epochs\t\t', args.max_epochs)
    print('epochs\t\t', args.epochs)
    print('lambda ce\t\t', args.lambda_ce)
    print('lambda hkd\t\t', args.lambda_hkd)
    print('lambda rkd\t\t', args.lambda_rkd)
    print('lambda fkd\t\t', args.lambda_fkd)
    print('lambda ft\t\t', args.lambda_ft)
    print('ft_beta\t\t', args.ft_beta)
    print('ce mode\t\t', args.ce_mode)
    print('nf\t\t', args.nf)
    print('max_ratio\t\t', args.max_ratio)
    print('ratio_inc\t\t', args.ratio_inc)
    print('eval_epoch\t\t', args.eval_epoch)
    print('\n')
    print('--- buffer parameter ---')
    print('init_samples\t\t', args.init_samples)
    print('per_task_samples\t\t', args.per_task_samples)
    print('gen_batch_size\t\t', args.gen_batch_size)
    print('save_data\t\t', args.save_data)
    print('use_unslt\t\t', args.use_unslt)
    print('feat_type\t\t', args.feat_type)
    print('\n')
    print('--- contrastive parameter ---')
    print('cont_lr\t\t', args.cont_lr)
    print('cont_epoch\t\t', args.cont_epoch)
    print('cont_blocks\t\t', args.cont_blocks)
    print('cont_step\t\t', args.cont_step)
    print('cont_rslt\t\t', args.cont_rslt)
    print('act\t\t', args.act)
    print('cont_temp\t\t', args.cont_temp)
    print('\n')
    print('--- inversion parameter ---')
    print('inversion lr\t\t', args.inversion_lr)
    print('train steps\t\t', args.train_steps)
    print('tune_steps\t\t', args.tune_steps)
    print('tune_lr\t\t', args.tune_lr)
    print('alpha pr\t\t', args.alpha_pr)
    print('alpha rf\t\t', args.alpha_rf)
    print('rf_factor\t\t', args.rf_factor)
    print('flip rate\t\t', args.flip_rate)
    print('layer_wise\t\t', args.layer_wise)
    print('layer_batch\t\t', args.layer_batch)
    print('boost_factor\t\t', args.boost_factor)
    print('search_param\t\t', args.search_param)
    print('inv_milestones\t\t', args.inv_milestones)
    print('inv_lr_rate\t\t', args.inv_lr_rate)
    print('inv_warmup\t\t', args.inv_warmup)
    print('layer_schedule\t\t', args.layer_schedule)
    print('\n')

    utils.set_random_seed(seed=args.seed)
    main(opts=args)
