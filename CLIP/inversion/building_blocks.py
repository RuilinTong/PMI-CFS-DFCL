# -*-coding:utf8-*-

import torch


class PreTransformBlock(torch.nn.Module):
    def __init__(self, class_embedding, positional_embedding, width, ln_pre, vpt=None):
        super().__init__()
        self.class_embedding = class_embedding
        self.positional_embedding = positional_embedding
        self.width = width
        self.ln_pre = ln_pre
        self.vpt = vpt

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if self.vpt is not None:
            visual_ctx = self.vpt.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        return x


class PostTransformBlock(torch.nn.Module):
    def __init__(self, proj, ln_post, normalize=False):
        super().__init__()
        self.proj = proj
        self.ln_post = ln_post
        self.normalize = normalize

    def forward(self, x):
        x_1 = x.permute(1, 0, 2)  # LND -> NLD
        x_2 = self.ln_post(x_1[:, 0, :])
        x_3 = x_2 @ self.proj
        if self.normalize:
            x_4 = x_3 / (x_3.norm(dim=-1, keepdim=True) + 1e-6)
        else:
            x_4 = x_3
        return x_4


class ResTransformerPromptBlock(torch.nn.Module):
    def __init__(self, prompt_learner, transformer_block, block_id):
        super().__init__()
        self.prompt_learner = prompt_learner
        self.transformer_block = transformer_block
        self.block_id = block_id
        self.query = None
        self.p_list = None

    def set_query(self, query):
        self.query = query
        # compute prompt
        self.p_list, _, _ = self.prompt_learner.forward(self.query, self.block_id, None, train=False, task_id=None)

    def forward(self, x):
        # compute output
        x = self.transformer_block(x, layer=self.block_id, prompt=self.p_list)
        return x


class PostTransformBlockPrompt(torch.nn.Module):
    def __init__(self, ln_post):
        super().__init__()
        self.ln_post = ln_post

    def forward(self, x):
        x_1 = x.permute(1, 0, 2)  # LND -> NLD
        x_2 = self.ln_post(x_1[:, 0, :])
        return x_2
