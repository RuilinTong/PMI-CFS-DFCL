import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
# from .vit import VisionTransformer
import json
import numpy as np
import copy
from functools import partial
from clip_backbones.myclip import create_model, create_model_and_transforms

# note - ortho init has not been found to help l2p/dual prompt
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

      
def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param 

def ortho_penalty(t, pt):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()


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
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        # self.ZS_clip_encode_text = clip_model.encode_text.to("cuda")
        self.memory_size = args['memory_size']
        self.args = args
        # self.n_tasks = 10
        self.n_tasks = int(args['n_classes']/args['increment'])
        self.task_count = -1
        self.tokenized_prompts = None

        self._init_smart(args)
        # self.prompt_template = load_json('utils/templates.json')[args['dataset']]
        self.prompt_template = ['a photo of a {}.']
        
        # text prompt init
        for i in self.t_layers:
            if i == 0 and args['ctx_init']:
                self.prompt_prefix = args['ctx_init']
                p = self.ctx_init(args['ctx_init'], args['n_ctx_text'])
            else:
                p = tensor_prompt(self.t_pool_size, self.t_p_length, self.t_emb_d)
            setattr(self, f'text_p_{i}',p.type(self.dtype))

        # visual prompt init
        for i in self.v_layers:
            p = tensor_prompt(self.v_pool_size, self.v_p_length, self.v_emb_d)
            k = tensor_prompt(self.v_pool_size, self.key_d)
            setattr(self, f'visual_p_{i}',p.type(self.dtype))
            setattr(self, f'visual_k_{i}',k.type(self.dtype))
            

    def _init_smart(self, args):
        # text prompt hyperparameters
        self.t_pool_size = args['n_classes']
        # self.t_pool_size = 1
        self.t_p_length = args['n_ctx_text']
        self.t_layers = np.arange(args['prompt_depth_text'])
        
        # visual prompt hyperparameters
        self.v_pool_size = args['pool_size']
        self.v_p_length = args['n_ctx_vision']
        self.v_layers = np.arange(args['prompt_depth_vision'])

    def ctx_init(self, ctx_init, n_ctx):
        ctx_init = ctx_init.replace("_", " ")
        prompt = self.tokenizer(ctx_init)
        with torch.no_grad():
            embedding = self.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        ctx_vectors = ctx_vectors.expand(self.t_pool_size, -1, -1)
        return torch.nn.Parameter(ctx_vectors, requires_grad = True)
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        if len(ctx.shape) == 2:
            ctx = ctx.unsqueeze(0).expand(prefix.shape[0], -1, -1)
        elif ctx.shape[0] == 1:
            ctx = ctx.expand(prefix.shape[0], -1, -1)
        
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)·
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    
    def increment_class(self, classnames, known_classes=None, total_classes=None):
        # rebuilding token embeddings
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        
        if known_classes > 0:
            p_size = self.tokenized_prompts.shape
            tokenized_prompts[:p_size[0], :p_size[1]] = self.tokenized_prompts
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.t_p_length:, :])  # CLS, EOS
        print(f"after expanding, token_prefix.shape = {self.token_prefix.shape}, token_suffix.shape = {self.token_suffix.shape}")
        
        self.n_cls = len(classnames)
        self.tokenized_prompts = tokenized_prompts          
            

    def text_prompts(self, x_query, l, known_classes=None, total_classes=None):

        # retrieve text prompts
        text_p = None
        if l in self.t_layers:
            p = getattr(self,f'text_p_{l}')
            
            s = known_classes
            f = total_classes    
            #############################################
            # freeze/control past tasks
            if not p.shape[0] == 1:
                if self.memory_size == 0:
                # if self.training:
                    if s > 0:
                        p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                    else:
                        p = p[s:f]
                else:
                    p = p[0:f]
            
 
            if l == 0:
                prefix = self.token_prefix
                suffix = self.token_suffix
                text_p = self.construct_prompts(p, prefix, suffix)
            else:
                text_p = p
        return text_p

    def image_prompts(self, x_query, l, known_classes=None, total_classes=None):
        # retrieve visual prompts
        visual_p = None
        if l in self.v_layers:
            K = getattr(self,f'visual_k_{l}')
            p = getattr(self,f'visual_p_{l}')
            
            # s = known_classes
            # f = total_classes            
            pt = int(self.v_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)  
            if self.memory_size == 0:
            # if self.training:
                if s > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                p = p[0:f]

            # with attention and cosine sim
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_query, dim=1)
            # print(q.shape, n_K.shape)
            w = q@n_K.T
            v_p = w.unsqueeze(-1).unsqueeze(-1) * p 
            visual_p = torch.sum(v_p, dim=1)
            
            # print(v_p.shape)
            
            loss = 0
        else:
            loss = 0
            
        return visual_p, loss
    
    def forward(self, x_query, l, known_classes=None, total_classes=None, text_layer = False):
        if text_layer:
            # print('generating text prompts')
            return self.text_prompts(x_query, l, known_classes=known_classes, total_classes=total_classes)
        else:
            return self.image_prompts(x_query, l, known_classes=known_classes, total_classes=total_classes)




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.transformer.get_cast_dtype()

    def forward(self, prompts, tokenized_prompts, x_query=None, prompt_learner=None, known_classes=None, total_classes=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _ = self.transformer(x, x_query=x_query, prompt_learner=prompt_learner, known_classes=known_classes, total_classes=total_classes)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, clip_model=clip_model, tokenizer=tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.zs_text_encoder = copy.deepcopy(clip_model.encode_text)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.transformer.get_cast_dtype()
        # self.tokenizer = tokenizer
        # self.total_epochs = cfg.epochs
        self.n_cls = len(classnames)
        self.feature_dim = clip_model.feature_dim
        
    def increment_class(self, classnames, known_classes=None, total_classes=None):
        self.n_cls = len(classnames)
        self.prompt_learner.increment_class(classnames, known_classes=known_classes, total_classes=total_classes)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def encode_image(self, image, known_classes=None, total_classes=None):
        x_query, _ = self.image_encoder(image)
        assert x_query is not None
        # print(x_query.shape)
        return self.image_encoder(image, x_query=x_query, prompt_learner=self.prompt_learner, known_classes=known_classes, total_classes=total_classes)
    
    def encode_text(self, prompts, tokenized_prompts, known_classes=None, total_classes=None):
        return self.text_encoder(prompts, tokenized_prompts, prompt_learner=self.prompt_learner, known_classes=known_classes, total_classes=total_classes)
    
    def forward(self, image, label=None, known_classes=None, total_classes=None, return_feat=False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner(x_query=None, l=0, known_classes=known_classes, total_classes=total_classes, text_layer=True)
        # Compute the prompted image and text features
        # text_features = self.text_encoder(prompts, tokenized_prompts)
        # image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.encode_text(prompts, tokenized_prompts, known_classes=known_classes, total_classes=total_classes)
        image_features, prompt_loss = self.encode_image(image.type(self.dtype), known_classes=known_classes, total_classes=total_classes)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        
        if self.training or return_feat:
            return image_features, text_features, self.logit_scale, prompt_loss
        else:
            # Compute the prompted logits
            logits = logit_scale * image_features @ text_features.t()
            return logits, logits.t()
        
        


def get_vl_prcil(cfg, n_classes):
    print(f"Loading CLIP (backbone: {cfg['backbone_type']}.{cfg['pretrained_weight']})")
    # clip_model = load_clip_to_cpu(cfg)
    design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0}
    
    clip_model, train_trfm, test_trfm, tokenizer = create_model_and_transforms(cfg['backbone_type'], pretrained=cfg['pretrained_weight'], design_details=design_details)
    # clip_model = load_clip_to_cpu(cfg).float()
    print(f"Total number of CLIP parameters: {sum(p.numel() for p in clip_model.parameters())}")
    print(f"Logit scale: {clip_model.logit_scale}")

    print("Building custom CLIP")
    model = CustomCLIP(cfg, n_classes, clip_model.eval(), tokenizer)

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