import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
# from .vit import VisionTransformer
import numpy as np
import copy
from functools import partial
from clip_backbones.myclip import create_model_and_transforms

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.empty(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.empty(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
        # nn.init.normal_(p, std=0.02)
    return p    


class PromptLearner(nn.Module):
    # def __init__(self, emb_d=768, key_dim=512, prompt_param=[10,20,6]):
    def __init__(self, emb_d=768, key_dim=768, prompt_param=[10,20,5]):
# class DualPrompt(nn.Module):
#     def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        # self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        # self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # # prompt locations
        # self.g_layers = [0,1]
        # self.e_layers = [2,3,4]
        self.g_layers = [0,1,2,3,4]
        self.e_layers = [0,1,2,3,4]
        # self.g_layers = [0]
        # self.e_layers = [0]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0]) # number of tasks

    # def process_task_count(self):
    #     self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:,task_id]).sum()
                    # loss = (1.0 - cos_sim[:,task_id]).mean()
                    # P_ = p[task_id].expand(len(x_querry),-1,-1)
                    P_ = p[task_id]
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).sum()
                    # loss = (1.0 - cos_sim[:,k_idx]).mean()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # select prompts
            if train and self.task_id_bootstrap:
                # i = int(self.e_p_length/2)
                # Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                # Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
                E_ = P_.reshape((-1,self.emb_d))
            else:
                # i = int(self.e_p_length/2)
                # Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                # Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
                E_ = P_.reshape((-1,self.emb_d))
                # assert 0
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            # j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            # P_ = p.expand(len(x_querry),-1,-1)
            # Gk = P_[:,:j,:]
            # Gv = P_[:,j:,:]
            G_ = p

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            # Pk = torch.cat((Ek, Gk), dim=1)
            # Pv = torch.cat((Ev, Gv), dim=1)
            # p_return = [Pk, Pv]
            # print(G_.shape, E_.shape)
            p_return = torch.cat((G_, E_), dim=0)
        elif e_valid:
            # p_return = [Ek, Ev]
            p_return = E_
        elif g_valid:
            # p_return = [Gk, Gv]
            p_return = P_
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


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
        
        self.head = nn.Linear(self.feature_dim, n_classes) if n_classes > 0 else nn.Identity()
        
        self.prompt = PromptLearner()

    # def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     if self.global_average_pool:
    #         return x.mean(dim=1), x
    #     else:
    #         return x[:, 0], x[:, 1:]

    def forward_feature(self, x: torch.Tensor, prompt=None, q=None, train=False, task_id=None):

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
                    p_list, loss, x = prompt.forward(q, i, x, train=True, task_id=task_id)
                    prompt_loss += loss
                else:
                    p_list, _, x = prompt.forward(q, i, x, train=False, task_id=task_id)
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

    def forward(self, x, pen=False, train=False, task_id=None):

        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.forward_feature(x)
                # print(q.shape)
                q = q[:,0,:]
            out, prompt_loss = self.forward_feature(x, prompt=self.prompt, q=q, train=train, task_id=task_id)
            out = out[:,0,:]
            # out = out[:, :25, :].mean(dim=1)
        else:
            assert 0
        #     out, _ = self.forward_feature(x)
        #     out = out[:,0,:]
        # out = out.view(out.size(0), -1)
        
        if not pen:
            out = self.head(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
    

def get_vl_dualprompt(cfg, n_classes):
    print(f"Loading CLIP (backbone: {cfg['backbone_type']}.{cfg['pretrained_weight']})")
    # clip_model = load_clip_to_cpu(cfg)
    design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0}
    
    clip_model, train_trfm, test_trfm, _ = create_model_and_transforms(cfg['backbone_type'], pretrained=cfg['pretrained_weight'], design_details=design_details)
    # clip_model = load_clip_to_cpu(cfg).float()
    
    print("Building custom CLIP")
    model = CustomViT(cfg, n_classes, clip_model.eval())

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