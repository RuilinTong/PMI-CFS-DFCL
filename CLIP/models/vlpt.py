import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from torchvision import transforms
from utils.toolkit import tensor2numpy, build_optimizer, cosine_lr
import copy

# from clip_backbones.myclip import create_model_and_transforms
from clip_backbones.vlpt_clip import get_vlpt
# from open_clip import get_tokenizer

def ema_update(model, ema_model, ema_decay):
    with torch.no_grad():
        for (name, param), (name_, ema_param) in zip(model.named_parameters(), ema_model.named_parameters()):
            if ema_param.requires_grad and "VPT" in name:
            # if ema_param.requires_grad:
                # print(f"EMA update the params of {name}")
                # ema_param.data = ema_decay * ema_param.data + (1.0 - ema_decay) * param.data
                ema_param = ema_decay * ema_param + (1.0 - ema_decay) * param


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        self._network, self.train_trfm, self.test_trfm = get_vlpt(args, [])
        self._network = self._network.to(self._device)
        self.train_trfm = transforms.Compose([  
            transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.test_trfm = self.train_trfm

        self.args = args
        self.epochs = args['tuned_epoch']
        self.batch_size = args['batch_size']
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
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
        
        self.class_names = data_manager.get_classnames(np.arange(0, self._total_classes))
        self._network.increment_class(self.class_names, known_classes=self._known_classes, total_classes=self._total_classes)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, tb_logger)
        if self._memory_size > 0:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module 
            
    def _train(self, train_loader, test_loader, tb_logger=None):
        self._network.to(self._device)
        
        # if self._cur_task > 0:
        #     logging.info("turning off gradients of vision prompts")
        #     for name, param in self._network.named_parameters():
        #         if "VPT" in name:
        #             param.requires_grad_(False)
        
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {enabled}")

        # optimizer = self.get_optimizer()
        # scheduler = self.get_scheduler(optimizer)
        optimizer = build_optimizer(self._network, self.args)
        scheduler = cosine_lr(
            optimizer, self.init_lr, 0, len(train_loader)*self.epochs
        )
            
        self._init_train(train_loader, test_loader, optimizer, scheduler, tb_logger)
        
    def _init_train(self, train_loader, test_loader, optimizer, scheduler, logger=None):
        
        if self._cur_task > 0:
                print("copied old model")
                old_model = copy.deepcopy(self._network)
          
        prog_bar = tqdm(range(self.args['tuned_epoch']))      
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            
            for iteration, (_, inputs, targets) in enumerate(train_loader, start=len(train_loader)*epoch):
                
                
                scheduler(iteration)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                            
                f_i, f_t, scale = self._network(inputs, known_classes=self._known_classes, total_classes=self._total_classes)
                ############################################################
                # CE loss
                logits = scale.exp() * f_i @ f_t.t()
                i_logits = logits
                if self._memory_size == 0:
                    i_logits[:, :self._known_classes] = float('-inf')
                loss_ce_i = F.cross_entropy(i_logits, targets.long())
                loss = loss_ce_i
                
                # anchor_labels = targets.contiguous().view(-1, 1)
                # contrast_labels = torch.arange(self._total_classes).view(-1,1).cuda()
                # mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda().t()
                # t_logits = scale.exp() * f_t @ f_i.t()
                # if self._memory_size == 0:
                #     t_logits = t_logits[self._known_classes:, :]
                #     mask = mask[self._known_classes:, :]
                # loss_ce_t = F.cross_entropy(t_logits, mask)
                
                # loss = (loss_ce_i + loss_ce_t)
                
                
                if logger:
                    total_iterations = len(train_loader) * self.epochs
                    logger.add_scalar("loss/train", loss, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss/ce_i", loss_ce_i, iteration + self._cur_task * total_iterations)
                    # logger.add_scalar("loss/ce_t", loss_ce_t, iteration + self._cur_task * total_iterations)
                    logger.add_scalar('Lr', optimizer.param_groups[0]['lr'], iteration + self._cur_task * total_iterations)
                ############################################################
                # Supervised contrastive loss
                # logits = scale.exp() * f_i @ f_t.t()
                # # logits = scale.exp() * f_i @ f_t.detach().t()
                
                # if self._memory_size == 0:
                #     logits[:, :self._known_classes] = float('-inf')
                # loss_ce_i = F.cross_entropy(logits, targets.long())
                
                # anchor_labels = targets.contiguous().view(-1, 1)
                # contrast_labels = torch.arange(self._total_classes).view(-1,1).cuda()
                # mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda().t()
                # if self._memory_size == 0:
                #     t_logits = logits.t()[self._known_classes:, :]
                #     mask = mask[self._known_classes: , :]
                # else:
                #     t_logits=logits.t()
                # # t_logits = scale.exp() * f_t @ f_i.detach().t()
                                    
                # neg = torch.log(torch.sum(torch.exp(t_logits), dim=1, keepdim=True))
                # log_prob = t_logits - neg
                # masked_log_prob = mask * log_prob
                # # avoid dividing by zero
                # masked_log_prob = torch.sum(masked_log_prob, dim=1)/(torch.sum(mask, dim=1)+1e-7)
                # loss_contra_t = -torch.mean(masked_log_prob)
                # loss = (loss_ce_i + loss_contra_t)
                # # loss = loss_ce_i
                
                # if logger:
                #     total_iterations = len(train_loader) * self.epochs
                #     logger.add_scalar("loss/train", loss, iteration + self._cur_task * total_iterations)
                #     logger.add_scalar("loss/ce_i", loss_ce_i, iteration + self._cur_task * total_iterations)
                #     logger.add_scalar("loss/contra_t", loss_contra_t, iteration + self._cur_task * total_iterations)
                #     logger.add_scalar('Lr', optimizer.param_groups[0]['lr'], iteration + self._cur_task * total_iterations)
                ############################################################

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
            # if self._cur_task > 0:
            #     print("EMA update the learnable params for each epoch")
            #     ema_update(old_model, self._network, 0.999)

            # if scheduler:
            #     scheduler.step()
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
            
        if self._cur_task > 0:
            print("EMA update the learnable params for each task")
            ema_update(old_model, self._network, 0.9999)
        
    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.image_encoder(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.image_encoder(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits, _ = self._network(inputs, known_classes=0, total_classes=self._total_classes)
            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

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

    #     return np.around(tensor2numpy(correct) * 100 / total, decimals=2)