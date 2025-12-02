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


class PromptLearner_Image(nn.Module):
    def __init__(self, args, v_emb_d=768, clip_model=None):
        super().__init__()
        self.v_emb_d = v_emb_d
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.logit_scale = clip_model.logit_scale
        # self.ZS_clip_encode_text = clip_model.encode_text.to("cuda")
        self.memory_size = args['memory_size']
        self.args = args
        
        self._init_smart(args)
        
        # visual prompt init
        for i in self.v_layers:
            p = tensor_prompt(self.v_p_length, self.v_emb_d)
            # k = tensor_prompt(self.v_pool_size, self.key_d)
            # a = tensor_prompt(self.v_pool_size, self.key_d)
            # a = torch.nn.Parameter(torch.ones(self.t_pool_size, self.key_d), requires_grad=True)
            setattr(self, f'visual_p_{i}',p.type(self.dtype))
            # setattr(self, f'visual_k_{i}',k.type(self.dtype))
            # setattr(self, f'visual_a_{i}',a.type(self.dtype))

    def _init_smart(self, args):        
        # visual prompt hyperparameters
        self.v_pool_size = args['n_classes']
        self.v_p_length = args['n_ctx_vision']
        self.v_layers = np.arange(args['prompt_depth_vision'])

    def image_prompts(self, l):
        # retrieve visual prompts
        visual_p = None
        if l in self.v_layers:
            # K = getattr(self,f'visual_k_{l}')
            # A = getattr(self,f'visual_a_{l}')
            p = getattr(self,f'visual_p_{l}')
            visual_p = p
                
        return visual_p
    
    def forward(self, l):
        return self.image_prompts(l)

class PromptLearner_Text(nn.Module):
    def __init__(self, args, t_emb_d=512, clip_model=None, tokenizer=None):
        super().__init__()
        self.t_emb_d = t_emb_d
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.tokenizer = tokenizer
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        # self.ZS_clip_encode_text = clip_model.encode_text.to("cuda")
        self.memory_size = args['memory_size']
        self.args = args
        
        self.tokenized_prompts = None

        self._init_smart(args)
        
        # text prompt init
        for i in self.t_layers:
            if i == 0 and args['ctx_init']:
                self.prompt_prefix = args['ctx_init']
                p = self.ctx_init(args['ctx_init'], args['n_ctx_text'])
            else:
                p = tensor_prompt(self.t_pool_size, self.t_p_length, self.t_emb_d)
            # k = tensor_prompt(self.t_pool_size, self.key_d)
            # a = tensor_prompt(self.t_pool_size, self.key_d)
            # a = torch.nn.Parameter(torch.ones(self.t_pool_size, self.key_d), requires_grad=True)
            setattr(self, f'text_p_{i}',p.type(self.dtype))
            # setattr(self, f'text_k_{i}',k.type(self.dtype))
            # setattr(self, f'text_a_{i}',a.type(self.dtype))            

    def _init_smart(self, args):
        # text prompt hyperparameters
        self.t_pool_size = args['n_classes']
        self.t_p_length = args['n_ctx_text']
        self.t_layers = np.arange(args['prompt_depth_text'])
        
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
            
               
    def text_prompts(self, l, known_classes=None, total_classes=None):

        # retrieve text prompts
        text_p = None
        if l in self.t_layers:
            # K = getattr(self,f'text_k_{l}')
            # A = getattr(self,f'text_a_{l}')
            p = getattr(self,f'text_p_{l}')
            
            s = known_classes
            f = total_classes            
            # freeze/control past tasks
            if self.memory_size == 0:
            # if self.training:
                if s > 0:
                    # K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    # A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    # K = K[s:f]
                    # A = A[s:f]
                    p = p[s:f]
            else:
                # K = K[0:f]
                # A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            # a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            # n_K = nn.functional.normalize(K, dim=1)
            # q = nn.functional.normalize(a_querry, dim=2)
            # aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            # text_p = torch.einsum('bk,kld->bld', aq_k, p)
            # text_p = p
            if l == 0:
                prefix = self.token_prefix
                suffix = self.token_suffix
                text_p = self.construct_prompts(p, prefix, suffix)
            else:
                text_p = p
        return text_p

    
    def forward(self, query, l, known_classes=None, total_classes=None, text_layer=None):
        return self.text_prompts(l, known_classes=known_classes, total_classes=total_classes)




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.transformer.get_cast_dtype()

    def forward(self, prompts, tokenized_prompts, prompt_learner=None, known_classes=None, total_classes=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, prompt_learner=prompt_learner, known_classes=known_classes, total_classes=total_classes)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner_text = PromptLearner_Text(cfg, clip_model=clip_model, tokenizer=tokenizer)
        # self.prompt_learner_image = PromptLearner_Image(cfg, clip_model=clip_model)
        self.tokenized_prompts = self.prompt_learner_text.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.transformer.get_cast_dtype()
        # self.tokenizer = tokenizer
        # self.total_epochs = cfg.epochs
        self.n_cls = len(classnames)
        self.feature_dim = clip_model.feature_dim
        
    def increment_class(self, classnames, known_classes=None, total_classes=None):
        self.n_cls = len(classnames)
        self.prompt_learner_text.increment_class(classnames, known_classes=known_classes, total_classes=total_classes)
        self.tokenized_prompts = self.prompt_learner_text.tokenized_prompts

    def encode_image(self, image):
        return self.image_encoder(image)
    
    def encode_text(self, prompts, tokenized_prompts, known_classes=None, total_classes=None):
        return self.text_encoder(prompts, tokenized_prompts, prompt_learner=self.prompt_learner_text, known_classes=known_classes, total_classes=total_classes)
    
    def forward(self, image, label=None, known_classes=None, total_classes=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner_text(None, l=0, known_classes=known_classes, total_classes=total_classes)
        # Compute the prompted image and text features
        # text_features = self.text_encoder(prompts, tokenized_prompts)
        # image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.encode_text(prompts, tokenized_prompts, known_classes=known_classes, total_classes=total_classes)
        image_features = self.encode_image(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if self.training:
            return image_features, text_features, self.logit_scale
        else:
            # Compute the prompted logits
            logits = logit_scale * image_features @ text_features.t()
            return logits, logits.t()
        
        # if self.prompt_learner.training:
        #     # Now calculate the frozen pre-trained features
        #     # fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
        #     # fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        #     with torch.no_grad():
        #         zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
        #         zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
        #         # Compute pre-trained frozen visual features
        #         # zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
        #         zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.cuda().t()

        #     return logits, text_features, fixed_embeddings, zero_shot_features, \
        #            image_features, zero_shot_logits
        # else:
        # return logits, logits.t()
        # return logits, logits.t()
        


def get_clcoop(cfg, n_classes):
    print(f"Loading CLIP (backbone: {cfg['backbone_type']}.{cfg['pretrained_weight']})")
    # clip_model = load_clip_to_cpu(cfg)
    design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg['prompt_depth_vision'],
                          "language_depth": 0,
                          "vision_ctx": cfg['n_ctx_vision'],
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
    
    return model, train_trfm, test_trfm