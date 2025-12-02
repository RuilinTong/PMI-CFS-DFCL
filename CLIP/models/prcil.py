import logging
import os
import numpy as np
import torch
import pickle
import json
from torch import nn
from tqdm import tqdm
from torch import optim
import time
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, build_optimizer, cosine_lr

# from clip_backbones.myclip import create_model_and_transforms
from clip_backbones.prcil_clip import get_vl_prcil
# from open_clip import get_tokenizer

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param      


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        self._network, self.train_trfm, self.test_trfm, self.tokenizer = get_vl_prcil(args, [])
        self._network = self._network.to(self._device)
        self.prompt_template = load_json('utils/templates.json')[args['dataset']]
        self.alpha_div = args['alpha_div']

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
        
        # self.measure_diversity(self.test_loader)

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

        # if self._cur_task == 0:
        #     self._total_classes_t0 = self._total_classes
        #     test_dataset_t0 = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test", trfm=self.train_trfm)
        #     self.test_loader_t0 = DataLoader(test_dataset_t0, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        
        self.class_names = data_manager.get_classnames(np.arange(0, self._total_classes))
        self._network.increment_class(self.class_names, known_classes=self._known_classes, total_classes=self._total_classes)
        self._network.prompt_learner.task_count += 1 
        text_features = []
        with torch.no_grad():
            for l in self.class_names:
                texts = [t.format(l) for t in self.prompt_template]
                texts = self.tokenizer(texts)
                class_embeddings = self._network.zs_text_encoder(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            self.text_features = torch.stack(text_features, dim=0).cuda()


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

        optimizer = build_optimizer(self._network, self.args)
        scheduler = cosine_lr(
            optimizer, self.args['init_lr'], 0, len(train_loader)*self.epochs
        )    
        self._init_train(train_loader, test_loader, optimizer, scheduler, tb_logger)
        
            
    def _init_train(self, train_loader, test_loader, optimizer, scheduler, logger=None):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for iteration, (_, inputs, targets) in enumerate(train_loader, start=len(train_loader)*epoch):
                start_time = time.time()
                scheduler(iteration)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                            
                f_i, f_t, scale, prompt_loss = self._network(inputs, known_classes=self._known_classes, total_classes=self._total_classes)                
                ############################################################
                # CE loss
                # logits = scale.exp() * f_i @ f_t.detach().t()
                logits = scale.exp() * f_i @ f_t.t()
                i_logits = logits
                if self._memory_size == 0:
                # if True:
                    i_logits[:, :self._known_classes] = float('-inf')
                loss_ce_i = F.cross_entropy(i_logits, targets.long())
                # loss = loss_ce_i
                
                anchor_labels = targets.contiguous().view(-1, 1)
                contrast_labels = torch.arange(self._total_classes).view(-1,1).cuda()
                mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda().t()
                # t_logits = scale.exp() * f_t @ f_i.detach().t()
                t_logits = scale.exp() * f_t @ f_i.t()
                if self._memory_size == 0:
                # if True:
                    t_logits = t_logits[self._known_classes:, :]
                    mask = mask[self._known_classes:, :]
                loss_ce_t = F.cross_entropy(t_logits, mask)
                
                # loss_t_div = (-F.pdist(f_t, p=2)).pow(2.0).mean()
                loss_t_div = F.pdist(f_t, p=2).pow(2).mul(-2).exp().mean()
                
                loss = (loss_ce_i + loss_ce_t + self.alpha_div * loss_t_div)
                
                if logger:
                    total_iterations = len(train_loader) * self.epochs
                    logger.add_scalar("loss_img/train", loss, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss_img/ce_i", loss_ce_i, iteration + self._cur_task * total_iterations)
                    # logger.add_scalar("loss_img/ortho_p", prompt_loss, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss_img/t_div", loss_t_div, iteration + self._cur_task * total_iterations)
                    # logger.add_scalar("loss_img/t_kd", loss_t_kd, iteration + self._cur_task * total_iterations)
                    logger.add_scalar("loss_img/ce_t", loss_ce_t, iteration + self._cur_task * total_iterations)
                    logger.add_scalar('Lr_img', optimizer.param_groups[0]['lr'], iteration + self._cur_task * total_iterations)
                ############################################################

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

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            # prog_bar.set_description(info)

            logging.info(info)
        
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
                    f, _ = self._network.image_encoder(_inputs.to(self._device))
                    _vectors = tensor2numpy(
                        f
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            start_time = time.time()
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
            end_time = time.time()
            elapsed_time = end_time - start_time
            # logging.info(f"testing one batch: {elapsed_time} seconds, {elapsed_time/inputs.shape[0]} seconds per image")
            # break

            
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def measure_diversity(self, loader):
        self._network.eval()
        v_features, zs_v_features, v_targets = [], [], []
        prompted_t_features = None
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features, text_features, _, _ = self._network(inputs, known_classes=0, total_classes=self._total_classes, return_feat=True)
                zs_img_features, _ = self._network.image_encoder(inputs)
                v_features.append(image_features.cpu().numpy())
                v_targets.append(targets.cpu().numpy())
                zs_v_features.append(zs_img_features.cpu().numpy())
                prompted_t_features = text_features
        
        v_features = np.concatenate(v_features)
        v_targets = np.concatenate(v_targets)  
        zs_v_features = np.concatenate((zs_v_features))
                
        outputs = {
            'v_features': v_features,
            'v_targets': v_targets,
            't_features': prompted_t_features,
            'zs_t_features': self.text_features,
            'zs_v_features': zs_v_features
        }
        
        save_path = os.path.join('weights', self.args['dataset'], 'prcil_w_t_div', f'diversity_t{self._cur_task}.pkl')
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'wb') as f:
            pickle.dump(outputs, f)
        
        


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