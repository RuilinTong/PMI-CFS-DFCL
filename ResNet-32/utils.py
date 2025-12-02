# -*-coding:utf8-*-

import torch
import random
import numpy as np

from backbone import resnet_cifar
from backbone import resnets
from backbone import classifier
from backbone import resnet_deep_inversion


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tv = torch.__version__
    if tv[:3] == '1.7' or tv[:3] == '1.8':
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(d=True)
    elif tv[:4] == '1.10' or tv[:4] == '1.13':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        pass


def build_model(dataset, bias=False, nf=16):
    if dataset == 'seq_cifar100' or dataset == 'seq_tinyimagenet':
        model = resnet_cifar.resnet32(nf=nf)
        head = classifier.DynamicSimpleHead(
            num_features=model.num_features, bias=bias
        )
    elif dataset == 'seq_imagenet100':
        model = resnets.resnet18()
        head = classifier.DynamicSimpleHead(
            num_features=model.num_features, bias=bias
        )
    else:
        raise ValueError('Invalid dataset')
    return model, head


def build_resnet(model_type, num_class):
    if model_type == 'resnet34':
        model = resnet_deep_inversion.ResNet34(num_classes=num_class)
    elif model_type == 'resnet18':
        model = resnet_deep_inversion.ResNet18(num_classes=num_class)
    else:
        raise ValueError('Invalid model type')
    head = torch.nn.Identity()
    return model, head


def check_gpu_memory():
    tv = torch.__version__
    if tv[:4] == '1.13':
        mem_inf = torch.cuda.mem_get_info()
    elif tv[:4] == '1.10':
        mem_inf = torch.cuda.memory_allocated()
    else:
        mem_inf = None
    return mem_inf
