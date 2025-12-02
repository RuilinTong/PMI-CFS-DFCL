import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
import time
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, build_optimizer, cosine_lr
from clip_backbones.vit_dap import get_vl_dap

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 1

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network, self.train_trfm, self.test_trfm = get_vl_dap(args, args['n_classes'])

        self.batch_size = args["batch_size"]
        self.epochs = args['tuned_epoch']
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.args = args
        self.sim_lambda = args["SIM_LAMBDA"]
        
        # total_params = sum(p.numel() for p in self._network.parameters())
        # logging.info(f'{total_params:,} model total parameters.')
        # total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        # logging.info(f'{total_trainable_params:,} model training parameters.')

        # # if some parameters are trainable, print the key name and corresponding parameter number
        # if total_params != total_trainable_params:
        #     for name, param in self._network.named_parameters():
        #         if param.requires_grad:
        #             logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager, tb_logger=None):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._memory_size > 0:
            logging.info(f"Rehearsal mode: training with memory size {self._memory_size}, fixed memory {self._fixed_memory}, {self._memory_per_class} per class")
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", appendent=self._get_memory(), trfm=self.train_trfm)
        else:
            logging.info("Rehearsal free mode")
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", trfm=self.train_trfm)
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        # test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test", trfm=self.test_trfm)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test", trfm=self.train_trfm)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        if self._cur_task == 1:
            for name, param in self._network.named_parameters():
                if "dap_downsample" in name :
                    param.requires_grad_(False)


        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, tb_logger)
        if self._memory_size > 0:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, tb_logger):
        self._network.to(self._device)
        
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {enabled}")

        optimizer = build_optimizer(self._network, self.args)
        scheduler = cosine_lr(
            optimizer, self.init_lr, 0, len(train_loader)*self.epochs
        )
            
        # if self._cur_task > 0:
        #     self._init_prompt(optimizer)

        # if self._cur_task > 0 and self.args["reinit_optimizer"]:
        #     optimizer = self.get_optimizer()
            
        self._init_train(train_loader, test_loader, optimizer, scheduler, tb_logger)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, logger=None):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            # self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            if_print=False
            for iteration, (_, inputs, targets) in enumerate(train_loader, start=len(train_loader)*epoch):
                start_time = time.time()
                scheduler(iteration)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output, prompt_loss = self._network(inputs, task_id=self._cur_task, train=True, if_print=if_print)
                
                # output, _ = self._network(inputs, train=True)
                logits = output[:, :self._total_classes]
                if self._memory_size == 0:
                    logits[:, :self._known_classes] = float('-inf')

                loss_ce = F.cross_entropy(logits, targets.long())
                # loss = loss_ce + prompt_loss
                loss = loss_ce - self.sim_lambda * prompt_loss
                
                if if_print:
                    print(targets[0])
                    print(loss_ce, prompt_loss)
                if_print=False
                # if self.args["pull_constraint"] and 'reduce_sim' in output:
                #     loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']

                if logger:
                    total_iterations = len(train_loader) * self.epochs
                    logger.add_scalar("loss/train", loss, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss/ce", loss_ce, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss/prompt_loss", prompt_loss, iteration + self._cur_task * total_iterations)
                    logger.add_scalar('Lr', optimizer.param_groups[0]['lr'], iteration + self._cur_task * total_iterations)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
            #     logging.info(f"training one batch: {elapsed_time} seconds, {elapsed_time/inputs.shape[0]} seconds per image")
            #     break
            # break


            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # if (epoch + 1) % 5 == 0:
            # test_acc = self._compute_accuracy(self._network, test_loader)
            # info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
            #     self._cur_task,
            #     epoch + 1,
            #     self.args['tuned_epoch'],
            #     losses / len(train_loader),
            #     train_acc,
            #     test_acc,
            # )
            # else:
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            # prog_bar.set_description(info)

            logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            start_time = time.time()
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=self._cur_task)[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            end_time = time.time()
            elapsed_time = end_time - start_time
            # logging.info(f"testing one batch: {elapsed_time} seconds, {elapsed_time/inputs.shape[0]} seconds per image")
            # break

            
        # print(np.concatenate(y_pred))

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors, _ = (
                        self._network.module.forward_feature(_inputs.to(self._device))
                    )
                else:
                    _vectors, _ = (
                        self._network.forward_feature(_inputs.to(self._device))
                    )
                _vectors = tensor2numpy(_vectors[:, 0, :].mean(dim=1).view(_vectors.size(0), -1))
                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    # def _compute_accuracy(self, model, loader):
    #     model.eval()
    #     correct, total = 0, 0
    #     for i, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
    #         predicts = torch.max(outputs, dim=1)[1]
    #         correct += (predicts.cpu() == targets).sum()
    #         total += len(targets)
    #     #     print(predicts)
    #     # print(self._total_classes)
        

    #     return np.around(tensor2numpy(correct) * 100 / total, decimals=2)