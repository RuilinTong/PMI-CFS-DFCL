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


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.empty(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.empty(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        # nn.init.uniform_(p)
        nn.init.normal_(p, std=0.02)
    return p    


def ortho_penalty(t, pt):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda())**2).mean()


class PromptLearner(nn.Module):
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
            K = getattr(self, f'visual_k_{l}')
            # A = getattr(self,f'visual_a_{l}')
            p = getattr(self, f'visual_p_{l}')
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
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    # A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    # A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                # A = A[0:f]
                p = p[0:f]

            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_query, dim=1)
            w = q@n_K.T

            if if_print:
                print(f"layer {l}: {w[0]}, {torch.sum(w[0])}")
            v_p = w.unsqueeze(-1).unsqueeze(-1) * p 
            visual_p = torch.sum(v_p, dim=1)
            
            loss = 0
        else:
            loss = 0
            
        return visual_p, loss
    
    def forward(self, x_query, l, known_classes=None, total_classes=None, text_layer=False, if_print=False, label=None):
        if text_layer:
            assert 0
            # print('generating text prompts')
            # return self.text_prompts(x_query, l, known_classes=known_classes, total_classes=total_classes)
        else:
            return self.image_prompts(x_query, l, known_classes=known_classes,
                                      total_classes=total_classes, if_print=if_print, label=label)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, clip_model=clip_model, tokenizer=tokenizer)
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.encode_text
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.transformer.get_cast_dtype()
        # self.total_epochs = cfg.epochs
        self.n_cls = len(classnames)
        self.feature_dim = clip_model.feature_dim
        
    def encode_text(self, text):
        return self.text_encoder(text)
        
    def encode_image(self, image, known_classes=None, total_classes=None, if_print=False, label=None):
        x_query, _ = self.image_encoder(image)
        assert x_query is not None
        # print(x_query.shape)
        image_features, prompt_loss = self.image_encoder(image, x_query=x_query, prompt_learner=self.prompt_learner, known_classes=known_classes, total_classes=total_classes, if_print=if_print, label=label)
        return image_features, prompt_loss, x_query

    def forward(self, image, text=None, label=None, known_classes=None, total_classes=None, if_print=False):
        # tokenized_prompts = self.tokenized_prompts
        # logit_scale = self.logit_scale.exp()

        # Compute the prompted image and text features
        if text is not None:
            text_features = self.text_encoder(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = None
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features, prompt_loss, zs_img = self.encode_image(image.type(self.dtype), known_classes=known_classes, total_classes=total_classes, if_print=if_print, label=label)
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
    design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0}
    
    clip_model, train_trfm, test_trfm, tokenizer = create_model_and_transforms(cfg['backbone_type'], pretrained=cfg['pretrained_weight'], design_details=design_details)
    logging.info(f"Total number of CLIP parameters: {sum(p.numel() for p in clip_model.parameters())}")
    
    print("Building custom CLIP")
    model = CustomCLIP(cfg, classnames, clip_model.cuda().eval(), tokenizer)

    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"

    for name, param in model.named_parameters():
        if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        else:
            if "token_embedding" in name:
                param.requires_grad_(False)
            if "ZS_image_encoder" in name:
                param.requires_grad_(False)
            if "ZS_clip" in name:
                param.requires_grad_(False)

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
