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
    def __init__(self, args, emb_d=768, key_dim=768, prompt_param=[100,5,0]):
# class DualPrompt(nn.Module):
#     def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.args = args
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        # self.n_tasks = n_tasks
        self.n_tasks = 10
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):
        # prompt_param = [100,8,0]
        # prompt basic param
        # self.e_pool_size = int(prompt_param[0]) #100
        # self.e_p_length = int(prompt_param[1])
        # self.e_layers = [0,1,2,3,4]
        
        self.e_pool_size = self.args['pool_size']
        self.e_p_length = self.args['n_ctx_vision']
        self.e_layers = range(self.args['prompt_depth_vision'])

        # strenth of ortho penalty
        # self.ortho_mu = prompt_param[2]
        self.ortho_mu = 0
        
    def process_task_count(self):
        self.task_count += 1
        if self.task_count == self.n_tasks:
            return

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block, train=False, task_id=None, if_print=False):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            if if_print:
                # print(aq_k.shape)
                print(aq_k[0], torch.sum(aq_k[0]))
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            # i = int(self.e_p_length/2)
            # Ek = P_[:,:i,:]
            # Ev = P_[:,i:,:]
            # E_ = P_

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            # p_return = [Ek, Ev]
            p_return = P_
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda())**2).mean()


class CustomViT(nn.Module):
    # output_tokens: torch.jit.Final[bool]

    def __init__(self, cfg, n_classes, clip_model):
        super().__init__()
        self.clip_model = clip_model
        vit = self.clip_model.visual
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
        self.embed_dim = 768
        self.feature_dim = 768
        # TODO: consider removing bias
        self.head = nn.Linear(self.feature_dim, n_classes) if n_classes > 0 else nn.Identity()
        
        self.prompt = PromptLearner(cfg)

    # def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     if self.global_average_pool:
    #         return x.mean(dim=1), x
    #     else:
    #         return x[:, 0], x[:, 1:]

    def forward_feature(self, x: torch.Tensor, prompt=None, q=None, train=False, task_id=None, if_print=False,
                        skip_conv=False):
        if not skip_conv:
            # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
            if self.input_patchnorm:
                # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
                x = x.reshape(
                    x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0],
                    self.grid_size[1], self.patch_size[1]
                )
                x = x.permute(0, 2, 4, 1, 3, 5)
                x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
                x = self.patchnorm_pre_ln(x)
                x = self.conv1(x)
            else:
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        else:
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
                    p_list, loss, x = prompt.forward(q, i, x, train=True, task_id=task_id, if_print=if_print)
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

    def forward(self, x, pen=False, train=False, task_id=None, if_print=False, skip_conv=False, norm_feat=False):

        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.forward_feature(x, skip_conv=skip_conv)
                # print(q.shape)
                q = q[:, 0, :]
            out, prompt_loss = self.forward_feature(
                x, prompt=self.prompt, q=q, train=train, task_id=task_id, if_print=if_print, skip_conv=skip_conv)
            out = out[:, 0, :]
            if norm_feat:
                out = F.normalize(out, dim=-1)
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
    

def get_vl_codaprompt(cfg, n_classes):
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
