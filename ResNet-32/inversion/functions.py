# -*-coding:utf8-*-

import torch
import torch.nn.functional as F
import math


class BNInputHook(object):
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.handle = self.module.register_forward_hook(hook=self.get_input_hook())

    def stats_regularization(self):
        # get bn layer stats:
        running_mean = self.module.running_mean.clone().detach()
        running_var = self.module.running_var.clone().detach()
        # get stats of input
        mean = self.inputs.mean([0, 2, 3])
        nch = self.inputs.shape[1]
        value = self.inputs.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8
        # compute KL divergence
        r = torch.log(var) - torch.log(running_var + 1e-8) - \
            (1 - (running_var + (mean - running_mean) ** 2) / var)
        r = r.mean() * 0.5
        return r

    def get_input_hook(self):

        def hook(module, input, output):
            self.inputs = input[0]

        return hook

    def remove_hook(self):
        self.inputs = None
        self.module = None
        self.handle.remove()


class ModifiedL2BNInputHook(object):
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.handle = self.module.register_forward_hook(hook=self.get_input_hook())
        self.loss_fn = torch.nn.MSELoss()

    def stats_regularization(self):
        # get bn layer stats:
        running_mean = self.module.running_mean.clone().detach()
        running_var = self.module.running_var.clone().detach()
        # get stats of input
        mean = self.inputs.mean([0, 2, 3])
        nch = self.inputs.shape[1]
        value = self.inputs.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8
        # compute KL divergence
        r_mean = self.loss_fn(mean, running_mean)
        r_var = self.loss_fn(var, running_var)
        r = r_mean + r_var
        return r

    def get_input_hook(self):

        def hook(module, input, output):
            self.inputs = input[0]

        return hook

    def remove_hook(self):
        self.inputs = None
        self.module = None
        self.handle.remove()


class ModifiedBNInputHook(object):
    """
    Fixed the problem of negative KL divergence.
    """
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.handle = self.module.register_forward_hook(hook=self.get_input_hook())

    def stats_regularization(self):
        # get bn layer stats:
        running_mean = self.module.running_mean.clone().detach()
        running_var = self.module.running_var.clone().detach()
        # get stats of input
        mean = self.inputs.mean([0, 2, 3])
        nch = self.inputs.shape[1]
        value = self.inputs.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8
        # compute KL divergence
        r = torch.log(running_var) - torch.log(var + 1e-8) - (1 - (var + (mean - running_mean) ** 2) / running_var)
        r = r.mean() * 0.5
        return r

    def get_input_hook(self):

        def hook(module, input, output):
            self.inputs = input[0]

        return hook

    def remove_hook(self):
        self.inputs = None
        self.module = None
        self.handle.remove()


class CustomBNInputHook(object):
    """
    Fixed the problem of negative KL divergence.
    """
    def __init__(self, module, running_mean, running_var):
        self.module = module
        self.inputs = None
        self.handle = self.module.register_forward_hook(hook=self.get_input_hook())
        self.running_mean = running_mean
        self.running_var = running_var

    def stats_regularization(self):
        # get bn layer stats:
        running_mean = self.running_mean
        running_var = self.running_var
        # get stats of input
        mean = self.inputs.mean([0, 2, 3])
        nch = self.inputs.shape[1]
        value = self.inputs.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8
        # compute KL divergence
        r = torch.log(running_var) - torch.log(var + 1e-8) - (1 - (var + (mean - running_mean) ** 2) / running_var)
        r = r.mean() * 0.5
        return r

    def get_input_hook(self):

        def hook(module, input, output):
            self.inputs = input[0]

        return hook

    def remove_hook(self):
        self.inputs = None
        self.module = None
        self.handle.remove()


class L2BNOutputHook(object):
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.handle = self.module.register_forward_hook(hook=self.get_input_hook())

    def l2_stats_regularization(self):
        running_mean = self.module.running_mean.clone().detach()
        running_var = self.module.running_var.clone().detach()
        # get stats of input
        mean = self.inputs.mean([0, 2, 3])
        nch = self.inputs.shape[1]
        value = self.inputs.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8
        # compute L2 stat regularization
        r_mean = torch.norm(running_mean - mean, 2)
        r_var = torch.norm(running_var - var, p=2)
        r = r_mean + r_var
        return r

    def get_input_hook(self):

        def hook(module, input, output):
            self.inputs = input[0]

        return hook

    def remove_hook(self):
        self.inputs = None
        self.module = None
        self.handle.remove()


class Gaussiansmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, inputs):
        """
        Apply gaussian filter to input.
        Arguments:
            inputs (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(inputs, weight=self.weight, groups=self.groups)


def get_bn_stats(model):
    stats = {}
    for n, mi in model.named_modules():
        if isinstance(mi, torch.nn.BatchNorm2d):
            mean = mi.running_mean.clone().detach()
            var = mi.running_var.clone().detach()
            stats[n] = ([mean, var])
    return stats


def freeze(model, mock_training=False):
    state = {}
    for name, param in model.named_parameters():
        state[name] = param.requires_grad
        param.requires_grad = False
        param.grad = None
    if mock_training and hasattr(model, "mock_training"):
        model.mock_training = True
    model.eval()
    return state


def unfreeze(model, state={}):
    default = None if state else True
    for name, param in model.named_parameters():
        requires_grad = state.get(name, default)
        if requires_grad is not None:
            param.requires_grad = requires_grad
    model.train()
    if hasattr(model, "mock_training"):
        model.mock_training = False


def denormalize(inputs, mean, var):
    mean = torch.unsqueeze(mean, dim=0)
    mean = torch.unsqueeze(mean, dim=2)
    mean = torch.unsqueeze(mean, dim=2)
    var = torch.unsqueeze(var, dim=0)
    var = torch.unsqueeze(var, dim=2)
    var = torch.unsqueeze(var, dim=2)
    denormed_inputs = inputs * var + mean
    return denormed_inputs
