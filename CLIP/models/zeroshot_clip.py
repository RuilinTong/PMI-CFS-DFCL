import logging
import json
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

from clip_backbones.myclip import create_model_and_transforms
from open_clip import get_tokenizer

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        design_details = {"trainer": 'IVLP',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0}
    
        self._network, self.train_trfm, self.test_trfm, self.tokenizer = create_model_and_transforms(args['backbone_type'], pretrained=args['pretrained_weight'], design_details=design_details)
        
        try:
            self.prompt_template = load_json('utils/templates.json')[args['dataset']]
        except:
            self.prompt_template = ["This is a photo of a {}."]
        self.text_tokens = None
        
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in self._network.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)
                if "ZS_clip" in name:
                    param.requires_grad_(False)
        self._network = self._network.to('cuda')

        self.args = args
        self.batch_size = args['batch_size']

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

        # if self._memory_size > 0:
        #     logging.info(f"Rehearsal mode: training with memory size {self._memory_size}, fixed memory {self._fixed_memory}, {self._memory_per_class} per class")
        #     train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", appendent=self._get_memory())
        # else:
        #     logging.info("Rehearsal free mode")
        #     train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        # self.train_dataset = train_dataset
        self.data_manager = data_manager
        # self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        # test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test", trfm=self.test_trfm)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test", trfm=self.train_trfm)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        self.class_names = data_manager.get_classnames(np.arange(0, self._total_classes))
        # self.text_tokens = self.tokenizer(
        #     [self.prompt_template.format(c) for c in self.class_names]
        # ).to('cuda')

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        # self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module   

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        
        text_features = []
        with torch.no_grad():
            for l in self.class_names:
                texts = [t.format(l) for t in self.prompt_template]
                texts = self.tokenizer(texts).cuda()
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
            
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
                image_features, _, logit_scale = self._network(inputs, self.text_tokens)
                logits = logit_scale * image_features @ text_features.t()
            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)        

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)