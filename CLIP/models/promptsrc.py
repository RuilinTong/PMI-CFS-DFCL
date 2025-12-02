import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, build_optimizer, cosine_lr

# from clip_backbones.myclip import create_model_and_transforms
from clip_backbones.prompsrc_clip import get_promptsrc
# from open_clip import get_tokenizer


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        self._network, self.train_trfm, self.test_trfm = get_promptsrc(args, [])
        self.ctx_init = args['ctx_init']
        self.text_tokens = None
        self._network = self._network.to(self._device)

        self.args = args
        self.epochs = args['tuned_epoch']
        self.batch_size = args['batch_size']
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005

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
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test", trfm=self.test_trfm)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        self.class_names = data_manager.get_classnames(np.arange(0, self._total_classes))
        self._network.expand_prompts(self.class_names)

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
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for iteration, (_, inputs, targets) in enumerate(train_loader, start=len(train_loader)*epoch):
                scheduler(iteration)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                            
                logits_per_image, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
                zero_shot_logits = self._network(inputs)
                # loss_ce = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
                if self._memory_size == 0:
                    logits_per_image[:, :self._known_classes] = float('-inf')
                loss_ce = F.cross_entropy(logits_per_image, targets.long())
                # Calculate the L_SCL_text loss
                loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                        reduction='mean') 
                # Calculate the L_SCL_image loss
                loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                        reduction='mean') 
                # Now calculate L_SCL_logits
                L_SCL_logits = F.kl_div(
                    F.log_softmax(logits_per_image / 1, dim=1),
                    F.log_softmax(zero_shot_logits / 1, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (1 * 1) / logits_per_image.numel()
                # L_SCL = (10 * L_SCL_logits + loss_scl_text * 100 + loss_scl_image * 100)
                L_SCL = (L_SCL_logits + loss_scl_text * 25 + loss_scl_image * 10)
                loss = (loss_ce + L_SCL)
                
                if logger:
                    total_iterations = len(train_loader) * self.epochs
                    logger.add_scalar("loss/train", loss, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss/ce", loss_ce, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss/scl_img", loss_scl_image, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss/scl_text", loss_scl_text, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss/scl_logits", L_SCL_logits, iteration + self._cur_task * total_iterations)
                    logger.add_scalar('Lr', optimizer.param_groups[0]['lr'], iteration + self._cur_task * total_iterations)
                # loss = loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits_per_image, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )

            logging.info(info)
        
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits, _ = self._network(inputs)
            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
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