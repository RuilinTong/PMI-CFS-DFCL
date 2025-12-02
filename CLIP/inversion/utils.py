# -*-coding:utf8-*-

import torch
import random
import numpy as np

from inversion import building_blocks


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


def split_clip_blocks(model, normalize=False, split_cnn=False):
    width = model.conv1.out_channels
    # pre process
    pre_trans_block = building_blocks.PreTransformBlock(
        class_embedding=model.class_embedding,
        positional_embedding=model.positional_embedding,
        width=width, ln_pre=model.ln_pre, vpt=None if not model.VPT_shallow else model.VPT
    )
    if split_cnn:
        blocks = [model.conv1, pre_trans_block]
        batch_dim = [0, 0]
    else:
        blocks = [torch.nn.Sequential(model.conv1, pre_trans_block)]
        batch_dim = [0]
    # get all blocks
    for n, mi in model.named_modules():
        if 'transformer.resblocks.' in n and len(n.split('.')) == 3:
            blocks.append(mi)
            batch_dim.append(1)
    # post process
    post_trans_block = building_blocks.PostTransformBlock(
        proj=model.proj, ln_post=model.ln_post, normalize=normalize
    )
    blocks.append(post_trans_block)
    batch_dim.append(1)
    return blocks, batch_dim


def split_vit_blocks_prompt(model, prompt, normalize=False, split_cnn=False):
    width = model.conv1.out_channels
    # pre process
    pre_trans_block = building_blocks.PreTransformBlock(
        class_embedding=model.class_embedding,
        positional_embedding=model.positional_embedding,
        width=width, ln_pre=model.ln_pre, vpt=None if not model.VPT_shallow else model.VPT
    )
    if split_cnn:
        blocks = [model.conv1, pre_trans_block]
        batch_dim = [0, 0]
        need_prompt = [False, False]
    else:
        blocks = [torch.nn.Sequential(model.conv1, pre_trans_block)]
        batch_dim = [0]
        need_prompt = [False]
    # get all blocks
    for i, bi in enumerate(model.transformer.resblocks):
        resblock = building_blocks.ResTransformerPromptBlock(prompt_learner=prompt, transformer_block=bi, block_id=i)
        blocks.append(resblock)
        batch_dim.append(1)
        need_prompt.append(True)
    # post process
    post_trans_block = building_blocks.PostTransformBlockPrompt(ln_post=model.ln_post)
    blocks.append(post_trans_block)
    batch_dim.append(1)
    need_prompt.append(False)
    return blocks, batch_dim, need_prompt
