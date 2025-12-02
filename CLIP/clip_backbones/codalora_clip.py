import torch
import torch.nn as nn
import torch.nn.functional as F


# from clip import clip
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import get_tokenizer
# from open_clip.tokenizer import _tokenizer, tokenize
from .imagenet_templates import IMAGENET_TEMPLATES
from .myclip import create_model, create_model_and_transforms
import numpy as np
import logging
import copy
import math

class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
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

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        if task > -0.5:
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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

class LoRALearner(nn.Module):
    # vit-b-16
    def __init__(self, args, v_emb_d=768, t_emb_d=512, key_dim=512, clip_model=None, tokenizer=None):
    # vit-l-14
    # def __init__(self, args, v_emb_d=1024, t_emb_d=768, key_dim=768, clip_model=None, tokenizer=None):
        super().__init__()
        self.v_emb_d = v_emb_d
        self.t_emb_d = t_emb_d
        self.key_d = key_dim
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.tokenizer = tokenizer
        # self.token_embedding = clip_model.token_embedding
        # self.logit_scale = clip_model.logit_scale
        # self.ZS_clip_encode_text = clip_model.encode_text.to("cuda")
        self.memory_size = args['memory_size']
        # self.n_tasks = 10
        try:
            self.n_tasks = int(args['n_classes']/args['increment'])
        except:
            self.n_tasks = 1
        self.task_count = -1
        self.args = args
        
        # self.tokenized_prompts = None

        self._init_smart(args)
        
        # visual prompt init
        for i in self.v_layers:
            # gp = tensor_prompt(self.v_p_length, self.v_emb_d)
            p = tensor_prompt(self.v_pool_size, self.v_p_length, self.v_emb_d)
            k = tensor_prompt(self.v_pool_size, self.key_d)
            # a = tensor_prompt(self.v_pool_size, self.key_d)
            # a = torch.nn.Parameter(torch.ones(self.t_pool_size, self.key_d), requires_grad=True)
            # setattr(self, f'visual_gp_{i}',gp.type(self.dtype))
            setattr(self, f'visual_p_{i}',p.type(self.dtype))
            setattr(self, f'visual_k_{i}',k.type(self.dtype))
            # setattr(self, f'visual_a_{i}',a.type(self.dtype))
            

    def _init_smart(self, args):        
        # visual prompt hyperparameters
        # self.v_pool_size = args['n_classes']
        self.v_pool_size = args['pool_size']
        # self.v_pool_size = 100
        self.v_p_length = args['n_ctx_vision']
        self.v_layers = np.arange(args['prompt_depth_vision'])

    def image_prompts(self, x_query, l, known_classes=None, total_classes=None, if_print=False, label=None):
        # retrieve visual prompts
        visual_p = None
        if l in self.v_layers:
            K = getattr(self,f'visual_k_{l}')
            # A = getattr(self,f'visual_a_{l}')
            p = getattr(self,f'visual_p_{l}')
            # gp = getattr(self, f'visual_gp_{l}')
            
            # s = known_classes
            # f = total_classes    
            ##############################################
            pt = int(self.v_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)  
            # s = 0
            # f = self.v_pool_size      
            # freeze/control past tasks
            if self.memory_size == 0:
            # if self.training:
                if s > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    # A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    # A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                # A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            # a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            # q = nn.functional.normalize(a_querry, dim=2)
            q = nn.functional.normalize(x_query, dim=1)
            # print(q.shape, n_K.shape)
            # w = self.logit_scale.exp()*(q@n_K.T)
            w = q@n_K.T
            # temp=0.01
            # w = torch.exp(q@n_K.T/temp) + 1e-7
            ###############################################
            # if label is not None:
            #     w = torch.zeros_like(w)
            #     for i in range(w.shape[0]):
            #         w[i, label[i]] = 1
                
            ###############################################
            # top K aggregation instead of global aggregation
            # self.top_k = 10
            # values, indices = torch.topk(w, self.top_k, dim=1)
            # mask = torch.zeros_like(w)

            # # Filling the mask with the top k values
            # for i in range(values.size(0)):
            #     for j in range(values.size(1)):
            #         mask[i, indices[i, j]] = values[i, j]
            # w = mask
            ###########################
            # if not self.training:
                # top 1 selection instead of global aggregation
            # self.top_k = 1
            # values, indices = torch.topk(w, self.top_k, dim=1)
            # mask = torch.zeros_like(w)

            # # Filling the mask with the top k values
            # for i in range(values.size(0)):
            #     for j in range(values.size(1)):
            #         mask[i, indices[i, j]] = values[i, j]
            # w = mask
                
                # print(f"testing: layer {l}: {w[0]}")
            ###########################
            # w = F.sigmoid(w)
            # w = F.softmax(w/0.05, dim=1)
            if if_print:
                print(f"layer {l}: {w[0]}, {torch.sum(w[0])}")
            v_p = w.unsqueeze(-1).unsqueeze(-1) * p 
            visual_p = torch.sum(v_p, dim=1)
            # visual_p = torch.concat([gp.unsqueeze(0).expand(visual_p.shape[0],-1,-1), visual_p], dim=1)
            # print(visual_p.shape)
            ##############################################
            # top K concatenation of prompts
            # self.top_k = 10
            # _, indices = torch.topk(w, self.top_k, dim=1)
            # visual_p = p[indices].reshape((x_query.shape[0], -1, self.v_emb_d))
            ##############################################
            
            # print(v_p.shape)
            if self.training:
                # loss = 0
                loss = ortho_penalty(K, pt) 
                loss += ortho_penalty(p.view(p.shape[0], -1), pt) 
            else:
                loss = 0
        else:
            loss = 0
            
        return visual_p, loss
    
    def forward(self, x_query, l, known_classes=None, total_classes=None, text_layer = False, if_print=False, label=None):
        if text_layer:
            assert 0
            # print('generating text prompts')
            # return self.text_prompts(x_query, l, known_classes=known_classes, total_classes=total_classes)
        else:
            return self.image_prompts(x_query, l, known_classes=known_classes, total_classes=total_classes, if_print=if_print, label=label)





class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner = None
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.encode_text
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.transformer.get_cast_dtype()
        # self.total_epochs = cfg.epochs
        self.n_cls = len(classnames)
        self.feature_dim = clip_model.feature_dim
                
    def encode_text(self, text):
        return self.text_encoder(text)
        
    def encode_image(self, image, known_classes=None, total_classes=None, if_print=False, label=None, task=-1):
        x_query, _ = self.image_encoder(image)
        x_query = x_query.detach()
        assert x_query is not None
        # print(x_query.shape)
        image_features, prompt_loss = self.image_encoder(image, x_query=x_query, prompt_learner=self.prompt_learner, known_classes=known_classes, total_classes=total_classes, if_print=if_print, label=label, task=task)
        return image_features, prompt_loss, x_query

        
    def forward(self, image, text=None, label=None, known_classes=None, total_classes=None, if_print=False, task=-1):
        # tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # Compute the prompted image and text features
        if text is not None:
            text_features = self.text_encoder(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = None
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features, prompt_loss, zs_img = self.encode_image(image.type(self.dtype), known_classes=known_classes, total_classes=total_classes, if_print=if_print, label=label, task=task)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        zs_img = zs_img / zs_img.norm(dim=-1, keepdim=True)
        
        # Compute the prompted logits
        # logits = logit_scale * image_features @ text_features.t()
        
        if self.training:
            return image_features, text_features, self.logit_scale, prompt_loss, zs_img
        else:
            return image_features, text_features, self.logit_scale
        # else:
        #     # Compute the prompted logits
        #     logits = logit_scale * image_features @ text_features.t()
        #     return logits, logits.t()
        # return logits, logits.t()
        

def get_codavpt(cfg, classnames):
    print(f"Loading CLIP (backbone: {cfg['backbone_type']}.{cfg['pretrained_weight']})")
    # clip_model = load_clip_to_cpu(cfg)
    design_details = {"trainer": 'LoRA',
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0,
                          "lora_depth_vision": cfg['lora_depth_vision'],
                          "lora_depth_text": 0,
                          "rank": cfg['rank'],}
    
    clip_model, train_trfm, test_trfm, tokenizer = create_model_and_transforms(cfg['backbone_type'], pretrained=cfg['pretrained_weight'], design_details=design_details)
    print(clip_model)
    logging.info(f"Total number of CLIP parameters: {sum(p.numel() for p in clip_model.parameters())}")
    
    print("Building custom CLIP")
    model = CustomCLIP(cfg, classnames, clip_model.cuda().eval(), tokenizer)
    
    print(model)

    print("Turning off gradients in both the image and the text encoder")

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
        # else:
        #     if "token_embedding" in name:
        #         param.requires_grad_(False)
        #     if "ZS_image_encoder" in name:
        #         param.requires_grad_(False)
        #     if "ZS_clip" in name:
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
    
    return model, train_trfm, test_trfm, tokenizer