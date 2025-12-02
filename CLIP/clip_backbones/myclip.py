import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import json
import logging
import os
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Callable, Sequence
from collections import OrderedDict

# from dataclasses import dataclass, asdict
from open_clip import get_tokenizer
from open_clip.pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
# from open_clip.model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
#     resize_pos_embed, get_cast_dtype
from open_clip.hf_model import HFTextEncoder
from open_clip.model import convert_weights_to_lp, get_cast_dtype, CLIPVisionCfg, CLIPTextCfg, convert_weights_to_fp16
from open_clip.modified_resnet import ModifiedResNet
from open_clip.transformer import LayerNormFp32, LayerNorm, QuickGELU, PatchDropout, AttentionalPooler,\
    ResidualAttentionBlock, LayerScale
# from open_clip.coca_model import CoCa
# from open_clip.loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss
from open_clip.openai import list_openai_models
from open_clip.pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from open_clip.transform import image_transform, AugmentationCfg
from open_clip.tokenizer import HFTokenizer, tokenize
from open_clip.factory import get_model_config, HF_HUB_PREFIX, list_models, load_state_dict, convert_to_custom_text_state_dict, resize_pos_embed
from open_clip.timm_model import TimmModel
from open_clip.utils import to_2tuple

from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
# from open_clip.model import convert_weights_to_lp, get_cast_dtype
from open_clip.pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url


class ResidualAttentionBlock_DAP(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            layer=None,
            prompt = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            add_prompt=False, text_layer=False, i=0, design_details=None     
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        # self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            layer=None,
            prompt = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        if (k_x is not None or v_x is not None):
            logging.info("ATTENTION!! PERFORMING Cross Attention")
            print("ATTENTION!! PERFORMING Cross Attention")

        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = q_x[0:q_x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand(q_x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                q_x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = q_x[:1, :, :]
                suffix = q_x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(q_x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                q_x = torch.cat([prefix, textual_context, suffix], dim=0)
                
            x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
            x = x + self.ls_2(self.mlp(self.ln_2(x)))
            return x

        elif prompt is not None and not self.text_layer:
            assert layer is not None
            # print("vision", layer, prompt.shape)
            if layer == 0:
                if len(prompt.shape) == 2:
                    visual_ctx = prompt.expand(q_x.shape[1], -1, -1).permute(1, 0, 2).half()
                elif len(prompt.shape) == 3:
                    visual_ctx = prompt.permute(1, 0, 2).half()
                else:
                    assert 0
                q_x = torch.cat([q_x, visual_ctx], dim=0)
            else:
                # Remove the outputs produced by learnable tokens of previous layer
                if len(prompt.shape) == 2:
                    prompt_length = prompt.shape[0]
                    # print("prompt length:", prompt_length) 
                    prefix = q_x[0:q_x.shape[0] - prompt_length, :, :] # we assume same prompt length across layers
                    # Create/configure learnable tokens of this layer
                    visual_context = prompt.expand(q_x.shape[1], -1, -1).permute(1, 0, 2).half()
                elif len(prompt.shape) == 3:
                    prompt_length = prompt.shape[1]
                    # print("visual prompt length:", prompt_length) 
                    prefix = q_x[0:q_x.shape[0] - prompt_length, :, :] # we assume same prompt length across layers
                    # Create/configure learnable tokens of this layer
                    visual_context = prompt.permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                q_x = torch.cat([prefix, visual_context], dim=0)
            x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
            x = x + self.ls_2(self.mlp(self.ln_2(x)))
            return x
        elif prompt is not None and self.text_layer:
            assert layer is not None
            # if layer == 0:
            #     if len(prompt.shape) == 2:
            #         visual_ctx = prompt.expand(q_x.shape[1], -1, -1).permute(1, 0, 2).half()
            #     elif len(prompt.shape) == 3:
            #         visual_ctx = prompt.permute(1, 0, 2).half()
            #     else:
            #         assert 0
            #     q_x = torch.cat([q_x, visual_ctx], dim=0)
            # else:
            # Appending the learnable tokens in different way
            # x -> [77, NCLS, DIM]
            # First remove the learnable tokens from previous layer
            if layer != 0:
                # print(layer, prompt.shape)
                if len(prompt.shape) == 2:
                    prompt_length = prompt.shape[0]
                    # print("textual prompt length:", prompt_length) 
                    # print("prompt length:", prompt_length) 
                    # we assume same prompt length across layers
                    prefix = q_x[:1, :, :]
                    suffix = q_x[1 + prompt_length:, :, :]
                    # Create/configure learnable tokens of this layer
                    textual_context = prompt.expand(q_x.shape[1], -1, -1).permute(1, 0, 2).half()
                elif len(prompt.shape) == 3:
                    prompt_length = prompt.shape[1]
                    # print("textual prompt length:", prompt_length) 
                    # we assume same prompt length across layers
                    prefix = q_x[:1, :, :]
                    suffix = q_x[1 + prompt_length:, :, :]
                    # Create/configure learnable tokens of this layer
                    if prompt.shape[0] == 1:
                        textual_context = prompt.expand(q_x.shape[1], -1, -1).permute(1, 0, 2).half()
                    else:
                        textual_context = prompt.permute(1, 0, 2).half()
            # Add the learnable tokens of this layer with the input, by replacing the previous
            # layer learnable tokens
            q_x = torch.cat([prefix, textual_context, suffix], dim=0)
            
            x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
            x = x + self.ls_2(self.mlp(self.ln_2(x)))
            return x       
        else:
            x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
            x = x + self.ls_2(self.mlp(self.ln_2(x)))
            return x


class Attention_LoRA(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
            r=64, n_tasks=10
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)
        
        # LoRA
        self.rank = r
        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0
        
        self.init_param()

    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def init_param_ada(self, t, r):
        self.lora_A_k[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_k[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)
        self.lora_A_v[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_v[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)

        nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k[t].weight)
        nn.init.zeros_(self.lora_B_v[t].weight)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, need_weights=False,
                # added for LoRA
                task=-1, register_hook=False, get_feat=False,get_cur_feat=False):
        # lora
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]
        
        L, N, C = x.shape
        print(x.shape)
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        
        ################## Lora
        # TODO: change to coda lora
        if task > -0.5:
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            # k = k + F.linear(x, weight_k).reshape(L, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            print(k.shape, F.linear(weight_k, k).shape)
            k = k + F.linear(x, weight_k).permute(2,0,1)
            # v = v + F.linear(x, weight_v).reshape(L, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + F.linear(x, weight_v).permute(2,0,1)
        #################

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x
    
    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
        return matrix_k, matrix_v
    
    def get_pre_matrix(self, task):
        with torch.no_grad():
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v


class ResidualAttentionBlock_LoRA(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            r=64
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = Attention_LoRA(d_model, n_head, r=r)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    # def attention(
    #         self,
    #         q_x: torch.Tensor,
    #         k_x: Optional[torch.Tensor] = None,
    #         v_x: Optional[torch.Tensor] = None,
    #         attn_mask: Optional[torch.Tensor] = None,
    #         task=-1
    # ):
    #     k_x = k_x if k_x is not None else q_x
    #     v_x = v_x if v_x is not None else q_x

    #     attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
    #     return self.attn(
    #         q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask, task=task
    #     )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            task=-1,
            layer=None,
            prompt = None,
            **kwargs
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        # x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask, task=task))
        x = q_x + self.ls_1(self.attn(self.ln_1(q_x), attn_mask=attn_mask, task=task))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            prompts_needed=0,
            text_layer=False,
            design_details = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.text_layer = text_layer

        # self.resblocks = nn.ModuleList([
        #     ResidualAttentionBlock(
        #         width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
        #     for _ in range(layers)
        # ])
        current_trainer = design_details['trainer']
        if current_trainer == 'IVLP' or current_trainer == 'VPT':
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_IVLP(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer,
                                            add_prompt=True, text_layer=text_layer, i=i, design_details=design_details)
                                             if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer,
                                            add_prompt=False, text_layer=text_layer, i=i, design_details=design_details)
                                             for i in range(layers)])
        elif current_trainer == 'DAP' and not text_layer:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_DAP(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer,)
                                            # add_prompt=True, text_layer=text_layer, i=i, design_details=design_details)
                                             if prompts_needed > i
                                             else ResidualAttentionBlock(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer,)
                                            # add_prompt=False, text_layer=text_layer, i=i, design_details=design_details)
                                             for i in range(layers)])
        elif current_trainer == "LoRA":
            if self.text_layer:
                lora_depth = design_details['lora_depth_text']
            else:
                lora_depth = design_details['lora_depth_vision']
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_LoRA(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer, r=design_details['rank'])
                                            if lora_depth > i
                                            else ResidualAttentionBlock(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer,)
                                             for i in range(layers)])
        # elif current_trainer == 'MaPLe':
        #     self.resblocks = nn.Sequential(
        #         *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer, i)
        #           for i in range(layers)])
        else:
            # Corresponds to default CoOp or CoCoOp
            # assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
            self.resblocks = nn.ModuleList([
                ResidualAttentionBlock(
                    width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
                for _ in range(layers)
            ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, x_query = None, prompt_learner=None, known_classes=None, total_classes=None, if_print=False, label=None, prompt_loss=None, task=-1):
        prompt_losses = []
        if prompt_loss is not None:
            prompt_losses.append(prompt_loss)
        if prompt_learner is not None:
            for i, blk in enumerate(self.resblocks):
                if i==0 and self.text_layer:
                    prompts = None # prompts for the first layer of text encoder are handeled by text encoder itself
                else:
                    try:
                        # prompts, prompt_loss = prompt_learner.forward(x_query, i, known_classes=known_classes, total_classes=total_classes, text_layer=self.text_layer, if_print=if_print)
                        prompts, prompt_loss = prompt_learner.forward(x_query, i, known_classes=known_classes, total_classes=total_classes, text_layer=self.text_layer)
                        prompt_losses.append(prompt_loss)
                    except:
                        prompts = prompt_learner.forward(x_query, i, known_classes=known_classes, total_classes=total_classes, text_layer=self.text_layer)
                    # try:
                    #     prompts = prompt_learner.forward(x_query, i, known_classes=known_classes, total_classes=total_classes, text_layer=self.text_layer, if_print=if_print, label=label)
                    # except:
                    #     prompts = prompt_learner.forward(x_query, i, known_classes=known_classes, total_classes=total_classes, text_layer=self.text_layer)
                    x = blk(x, layer=i, prompt=prompts, attn_mask=attn_mask)
        else:
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    # try:
                        # added for lora
                    # x = r(x, attn_mask=attn_mask, task=task)
                    # except:
                    x = r(x, attn_mask=attn_mask)
        return x, sum(prompt_losses)/len(prompt_losses) if len(prompt_losses) > 0 else 0


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            attentional_pool: bool = False,
            n_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            input_patchnorm: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            design_details=None
    ):
        super().__init__()
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # prompt embedding
        if design_details["vision_depth"] == 0 or design_details["trainer"] == 'DAP':
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
        
        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = norm_layer(width)
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            prompts_needed=self.prompt_till_layer_visual,
            text_layer=False,
            design_details=design_details
        )

        self.global_average_pool = global_average_pool
        if attentional_pool:
            self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
            self.ln_post = norm_layer(output_dim)
            self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward(self, x: torch.Tensor, x_query=None, prompt_learner=None, known_classes=None, total_classes=None, if_print=False, label=None, task=-1):

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            # assert 0
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
            # print(visual_ctx.shape, x.shape)
        else:
            assert self.prompt_till_layer_visual == 0
            
        # Normal code as before

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, prompt_loss = self.transformer(x, x_query=x_query, prompt_learner=prompt_learner, known_classes=known_classes, total_classes=total_classes, if_print=if_print, label=label, task=task)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled, prompt_loss


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            embed_cls: bool = False,
            pad_id: int = 0,
            output_tokens: bool = False,
            design_details = None
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        prompt_till_layer_text = design_details['language_depth']
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details
        )
        self.ln_final = norm_layer(width)

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        design_details = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            design_details=design_details
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        design_details = None
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
            design_details=design_details
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            design_details = None
    ):
        super().__init__()
                
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype, design_details=design_details)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype, design_details=design_details)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)
        self.feature_dim = embed_dim

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features, _ = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print(x.shape)
        x, _ = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]
    resize_pos_embed(state_dict, model)
    # incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    return incompatible_keys


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
        design_details = None
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
        design_details=design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    # model.load_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def load_openai_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[str] = None,
        design_details = None
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : Union[str, torch.device]
        The device to put the loaded model
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'

    try:
        if get_pretrained_url(name, 'openai'):
            model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")
    except:
        weight_path = os.path.join(os.getcwd(), 'weights')
        model_path = os.path.join(weight_path, 'ViT-B-16.pt')
        # model_path = "/g/data/kf26/hl3559/code/prcil/weights/ViT-B-16.pt"

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        state_dict = torch.load(model_path, map_location="cpu")

    # Build a non-jit model from the OpenAI jitted model state dict
    cast_dtype = get_cast_dtype(precision)
    try:
        model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype, design_details=design_details)
    except KeyError:
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype, design_details=design_details)

    # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
    model = model.to(device)
    # FIXME support pure fp16/bf16 precision modes
    if precision != 'fp16':
        model.float()
        if precision == 'bf16':
            # for bf16, convert back to low-precision
            convert_weights_to_lp(model, dtype=torch.bfloat16)

    # add mean / std attributes for consistency with OpenCLIP models
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD
    return model


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config_path = download_pretrained_from_hf(model_id, filename='open_clip_config.json', cache_dir=cache_dir)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        pretrained_cfg = config['preprocess_cfg']
        model_cfg = config['model_cfg']
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        pretrained_cfg = {}
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
            ##############################
            **model_kwargs
            ###############################
        )
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f'Loaded {model_name} model config.')
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        if pretrained_image:
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

        # if custom_text:
        #     if is_hf_model:
        #         model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf
        #     if "coca" in model_name:
        #         model = CoCa(**model_cfg, **model_kwargs, cast_dtype=cast_dtype)
        #     else:
        #         model = CustomTextCLIP(**model_cfg, **model_kwargs, cast_dtype=cast_dtype)
        # else:
        model = CLIP(**model_cfg, **model_kwargs, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            if is_timm_model:
                # FIXME this is a bit janky, create timm based model in low-precision and
                # then cast only LayerNormFp32 instances back to float32 so they don't break.
                # Why? The convert_weights_to_lp fn only works with native models.
                model.to(device=device, dtype=dtype)
                from open_clip.transformer import LayerNormFp32
                def _convert_ln(m):
                    if isinstance(m, LayerNormFp32):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)
                model.apply(_convert_ln)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                try:
                    checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
                except:
                    weight_path = os.path.join(os.getcwd(), 'weights')
                    checkpoint_path = os.path.join(weight_path, 'vit_b_16-laion400m_e32-55e67d44.pt')
                    # checkpoint_path = "/g/data/kf26/hl3559/code/prcil/weights/vit_b_16-laion400m_e32-55e67d44.pt"
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True
        elif has_hf_hub_prefix:
            logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    return model


def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        **model_kwargs,
):
    print(model_name)
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        **model_kwargs,
    )
    
    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )
    tokenizer = get_tokenizer(model_name)
    print(preprocess_train)

    return model, preprocess_train, preprocess_val, tokenizer


# design_details = {"trainer": 'IVLP',
#                 "vision_depth": 0,
#                 "language_depth": 0, "vision_ctx": 0,
#                 "language_ctx": 0}
# # design_details = {"trainer": 'IVLP',
# #                 "vision_depth": 5,
# #                 "language_depth": 5, "vision_ctx": 4,
# #                 "language_ctx": 4}

# # model, preprocess_train, preprocess_val = create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32', design_details=design_details)
# model, preprocess_train, preprocess_val = create_model_and_transforms('ViT-B-16', pretrained='openai', design_details=design_details)
# tokenizer = get_tokenizer('ViT-B-16')
# print(model)
# print(tokenizer)