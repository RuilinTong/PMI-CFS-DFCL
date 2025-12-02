import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from functools import reduce
import numpy as np
from operator import mul
from torch.nn.modules.utils import _pair
import math
import copy
from functools import partial
from torch.nn.modules.utils import _pair
from open_clip.transformer import LayerNorm
from clip_backbones.myclip import create_model_and_transforms



class PromptLearner(nn.Module):
    def __init__(self, args, emb_d=768, key_dim=768, prompt_param=[100,5,0]):
        super().__init__()
        self.args = args
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        # self.n_tasks = n_tasks
        self.n_tasks = 10
        
        self.pool_size = self.args['PROMPT_POOL']
        self.num_tasks_emb =  self.args['NUM_TASKS_FOR_EMB']
        self.cur_lambda = self.args['CURRENT_LAMBDA']
        self.sim_lambda = self.args['SIM_LAMBDA']
        self.task_emb = self.args['TASK_EMB']
        self.num_dap_tokens = self.args['NUM_DAP_TOKENS']
        
        # self.patch_size = _pair(197)
        # self.prompt_dim = emb_d
        # val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.prompt_dim))
        # self.dap_key_embeddings = nn.Parameter(torch.zeros(self.pool_size, self.prompt_dim))
        # nn.init.uniform_(self.dap_key_embeddings.data, -val, val)
        # self.dap_emb = torch.nn.Embedding(self.num_tasks_emb, self.task_emb)

        
        self.dap_downsample = nn.ModuleList([nn.Linear(197, self.num_dap_tokens) for _ in range(12)])
        for i in range(12):
            nn.init.zeros_(self.dap_downsample[i].weight) 
            nn.init.zeros_(self.dap_downsample[i].bias) 
        self.dap_film = nn.ModuleList([nn.Linear(self.task_emb, emb_d * 2) for _ in range(12)])
        self.dap_norm = nn.ModuleList([LayerNorm(emb_d, eps=1e-6) for _ in range(12)])


    def forward(self, x_querry, l, x_block, train=False, task_id=None, if_print=False, task_id_estimated_emb=None):
        # return None, 0, x_block
    
        if l == 0:
            x_norm = self.dap_norm[l](x_block)
            # x_tran = torch.transpose(x_norm, 2, 1)
            x_tran = x_norm.permute(1,2,0)
            down = self.dap_downsample[l](x_tran)

            film = self.dap_film[l](task_id_estimated_emb)
            gamma4 = film[:, :self.emb_d]
            beta4 = film[:, self.emb_d:]
            gamma_norm = gamma4.norm(p=2, dim=1, keepdim=True).detach()
            beta_norm = beta4.norm(p=2, dim=1, keepdim=True).detach()

            gamma4 = gamma4.div(gamma_norm).view(film.size(0), -1, 1)
            beta4 = beta4.div(beta_norm).view(film.size(0), -1, 1)
            down = gamma4 * down + beta4
            down = torch.transpose(down, 2, 1)
        else:    
            x = torch.cat((
                x_block[ :1, :, :],
                x_block[(1+self.num_dap_tokens):, :, :]
            ), dim=0)

            x_norm = self.dap_norm[l](x)
            # x_tran = torch.transpose(x_norm, 2, 1)
            x_tran = x_norm.permute(1,2,0)
            down = self.dap_downsample[l](x_tran)

            film = self.dap_film[l](task_id_estimated_emb)
            gamma4 = film[:, :self.emb_d]
            beta4 = film[:, self.emb_d:]
            gamma_norm = gamma4.norm(p=2, dim=1, keepdim=True).detach()
            beta_norm = beta4.norm(p=2, dim=1, keepdim=True).detach()

            gamma4 = gamma4.div(gamma_norm).view(film.size(0), -1, 1)
            beta4 = beta4.div(beta_norm).view(film.size(0), -1, 1)
            down = gamma4 * down + beta4
            down = torch.transpose(down, 2, 1)

            # if not (layer_index == 11 and cfg.DATA.NAME == 'imagenet_r'):
            # # for imagenet_r, do not append prompts on the last layer
            #     x = torch.cat((
            #         x[:, :1, :],
            #         down,
            #         x[:, 1:, :]
            #     ), dim=1)
        return down, 0, x_block
        
        
        

class CustomViT(nn.Module):
    # output_tokens: torch.jit.Final[bool]

    def __init__(self, cfg, n_classes, clip_model):
        super().__init__()
        vit = clip_model.visual
        self.input_patchnorm = vit.input_patchnorm
        self.grid_size = vit.grid_size
        self.patch_size = vit.patch_size
        self.patchnorm_pre_ln = vit.patchnorm_pre_ln
        self.conv1 = vit.conv1
        self.class_embedding = vit.class_embedding
        self.positional_embedding = vit.positional_embedding
        self.patch_dropout = vit.patch_dropout
        self.ln_pre = vit.ln_pre
        # self.transformer = vit.transformer
        self.transformer_blocks = vit.transformer.resblocks
        self.attn_pool = vit.attn_pool
        self.ln_post = vit.ln_post
        self._global_pool = vit._global_pool
        self.output_tokens = vit.output_tokens
        self.proj = vit.proj
        self.embed_dim=768
        self.feature_dim=768
        
        self.args = cfg
        self.pool_size = self.args['PROMPT_POOL']
        self.num_tasks_emb =  self.args['NUM_TASKS_FOR_EMB']
        self.cur_lambda = self.args['CURRENT_LAMBDA']
        self.sim_lambda = self.args['SIM_LAMBDA']
        self.task_emb = self.args['TASK_EMB']
        self.num_dap_tokens = self.args['NUM_DAP_TOKENS']

        self.top_k = 1
        self.patch_size = _pair(197)
        self.prompt_dim = self.embed_dim
        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.prompt_dim))
        self.dap_key_embeddings = nn.Parameter(torch.zeros(self.pool_size, self.prompt_dim))
        nn.init.uniform_(self.dap_key_embeddings.data, -val, val)
        self.dap_emb = torch.nn.Embedding(self.num_tasks_emb, self.task_emb)
        
        self.head = nn.Linear(self.feature_dim, n_classes) if n_classes > 0 else nn.Identity()
        
        self.prompt = PromptLearner(cfg)

    # def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     if self.global_average_pool:
    #         return x.mean(dim=1), x
    #     else:
    #         return x[:, 0], x[:, 1:]

    def forward_feature(self, x: torch.Tensor, prompt=None, q=None, train=False, task_id=None, if_print=False, task_id_estimated_emb=None):

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
        # if self.VPT_shallow:
        #     visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
        #     x = torch.cat([x, visual_ctx], dim=1)
        # else:
        #     assert self.prompt_till_layer_visual == 0
            
        # Normal code as before

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
        for i, blk in enumerate(self.transformer_blocks):

            if prompt is not None:
                if train:
                    p_list, loss, x = prompt.forward(q, i, x, train=True, task_id=task_id, if_print=if_print, task_id_estimated_emb=task_id_estimated_emb)
                    prompt_loss += loss
                else:
                    p_list, _, x = prompt.forward(q, i, x, train=False, task_id=task_id, task_id_estimated_emb=task_id_estimated_emb)
                # if p_list is not None and i == 1:
                #     print(x[0,0,0:10])
                #     print(p_list[0][0,0,0:10])
                #     print(apple)
                # if p_list is not None:
                #     x = torch.concat((x[:,0,:].unsqueeze(1),p_list[0],p_list[1],x[:,1:,:]), dim=1)
                #     p_list = None
            else:
                p_list = None

            x = blk(x, layer=i, prompt=p_list)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        return x, prompt_loss

        # if self.attn_pool is not None:
        #     x = self.attn_pool(x)
        #     x = self.ln_post(x)
        #     pooled, tokens = self._global_pool(x)
        # else:
        #     pooled, tokens = self._global_pool(x)
        #     pooled = self.ln_post(pooled)

        # if self.proj is not None:
        #     pooled = pooled @ self.proj

        # if self.output_tokens:
        #     return pooled, tokens
        
        # return pooled, prompt_loss

    def forward(self, x, pen=False, train=False, task_id=None, if_print=False):
        B = x.shape[0]
        with torch.no_grad():
            q, _ = self.forward_feature(x)
            q = q[:,0,:]
            x_cls_embed = q
        
        if self.training:
            start = task_id * self.top_k
            end = (task_id + 1) * self.top_k
            prompt_mask = torch.arange(start, end).cuda()
            if end > self.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None


        dap_prompt_key_norm = F.normalize(self.dap_key_embeddings, dim=-1)

        x_embed_norm = F.normalize(x_cls_embed, dim=-1)
        sim = torch.matmul(dap_prompt_key_norm,
                           torch.transpose(x_embed_norm, 1, 0))

        sim = torch.transpose(sim, 1, 0)
        (sim_top_k, idx) = torch.topk(sim, self.top_k)
        idx = idx.squeeze(dim=-1)

        prompt_id, id_counts = torch.unique(idx, return_counts=True)
        _, major_idx = torch.topk(id_counts, self.top_k)
        major_prompt_id = prompt_id[major_idx]
        idx = expand_to_batch(major_prompt_id, x_cls_embed.shape[0]).squeeze(dim=-1)

        task_id = major_prompt_id.cpu()[0]

        if prompt_mask is not None:
            idx = prompt_mask
            task_id = idx.cpu()[0]
            idx = expand_to_batch(idx, x_cls_embed.shape[0]).squeeze(dim=-1)

        task_id_estimated_emb = self.dap_emb(idx)

        i = torch.arange(B).reshape(B, 1, 1)
        l = torch.arange(self.prompt_dim).reshape(1, 1, self.prompt_dim)

        selected_prompt_key = dap_prompt_key_norm.repeat(B, 1, 1)[
            i, idx.unsqueeze(-1), l]

        x_embed_norm = x_embed_norm.unsqueeze(1)
        sim_pull = selected_prompt_key * x_embed_norm
        reduce_sim = torch.sum(sim_pull) / x_cls_embed.shape[0]
        ##################################################################################
        if self.prompt is not None:
            # with torch.no_grad():
            #     q, _ = self.forward_feature(x)
            #     # print(q.shape)
            #     q = q[:,0,:]
            out, _ = self.forward_feature(x, prompt=self.prompt, q=q, train=train, task_id=task_id, if_print=if_print, task_id_estimated_emb=task_id_estimated_emb)
            out = out[:,0,:]
            # out = out[:, :25, :].mean(dim=1)
        else:
            assert 0
                
        if not pen:
            out = self.head(out)
        if self.prompt is not None and train:
            return out, reduce_sim
        else:
            return out

def expand_to_batch(x, batch_size, dim=0):
    shape = [1 for _ in x.shape]
    shape.insert(dim, batch_size)
    return torch.tile(torch.unsqueeze(x, dim=dim), shape).cuda()


def get_vl_dap(cfg, n_classes):
    print(f"Loading CLIP (backbone: {cfg['backbone_type']}.{cfg['pretrained_weight']})")
    # clip_model = load_clip_to_cpu(cfg)
    # design_details = {"trainer": 'DAP',
    #                       "vision_depth": 0,
    #                       "language_depth": 0,
    #                       "vision_ctx": 0,
    #                       "language_ctx": 0}
    design_details = {"trainer": 'IVLP',
                        "vision_depth": 0,
                        "language_depth": 0,
                        "vision_ctx": 0,
                        "language_ctx": 0}
    
    clip_model, train_trfm, test_trfm, _ = create_model_and_transforms(cfg['backbone_type'], pretrained=cfg['pretrained_weight'], design_details=design_details)
    # clip_model = load_clip_to_cpu(cfg).float()
    
    print("Building custom CLIP")
    model = CustomViT(cfg, n_classes, clip_model.eval())
    print(model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if not ("prompt" in name or "head" in name) :
            param.requires_grad_(False)
        # else:
        #     if "token_embedding" in name:
        #         param.requires_grad_(False)


    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")
    print(f"Parameters count: {len(enabled)}")
    print(f"Total number of tunable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model = model.to('cuda')

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    
    return model, train_trfm, test_trfm