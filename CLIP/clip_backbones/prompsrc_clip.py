import torch
import torch.nn as nn


# from clip import clip
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from open_clip import get_tokenizer
# from open_clip.tokenizer import _tokenizer, tokenize
from .imagenet_templates import IMAGENET_TEMPLATES
from .myclip import create_model, create_model_and_transforms


_tokenizer = get_tokenizer('ViT-B-16')

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.transformer.get_cast_dtype()

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        # assert cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
        #                                                 "\nPlease use VPT trainer if you want to learn only vision " \
        #                                                 "branch"
        # n_ctx = cfg.TRAINER.PROMPTSRC.N_CTX_TEXT
        # ctx_init = cfg.TRAINER.PROMPTSRC.CTX_INIT
        assert cfg['prompt_depth_text'] >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        self.n_ctx = cfg['n_ctx_text']
        self.ctx_init = cfg['ctx_init']
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # # cfg_imsize = cfg.INPUT.SIZE[0]
        # cfg_imsize = 224
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        if self.ctx_init and self.n_ctx <= 4:
            # use given words to initialize context vectors
            self.ctx_init = self.ctx_init.replace("_", " ")
            n_ctx = self.n_ctx
            # prompt = clip.tokenize(self.ctx_init)
            prompt = _tokenizer(self.ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = self.ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        # print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTSRC.N_CTX_VISION}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg['n_ctx_vision']}")
        self.ctx = nn.Parameter(ctx_vectors)
        
        print(f"prompt learner.ctx: {self.ctx.shape}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        try:
            # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            tokenized_prompts = torch.cat([_tokenizer(p) for p in prompts])  # (n_cls, n_tkn)
        except:
            tokenized_prompts = torch.tensor([[]], dtype=torch.long)
            
        # Also create frozen CLIP
        # clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        # clip_model_temp_image = load_clip_to_cpu(cfg, True)
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        #     self.ZS_image_encoder = clip_model_temp_image.visual
        #     # Now pre-compute the frozen VL embeddings
        #     all_teacher_features = []
        #     # Using multiple text templates to ensure textual diversity during training
        #     for single_template in IMAGENET_TEMPLATES:
        #         x = [single_template.replace("{}", name) for name in classnames]
        #         x_tokenized = torch.cat([clip.tokenize(p) for p in x])
        #         text_features = clip_model_temp.encode_text(x_tokenized.cuda())
        #         all_teacher_features.append(text_features.unsqueeze(1))

        # self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        
        #################################################################################
        # clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        # clip_model_temp_image = load_clip_to_cpu(cfg, True)
        design_details = {"trainer": 'IVLP',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0}
        clip_model_temp = create_model(cfg['backbone_type'], pretrained=cfg['pretrained_weight'], design_details = design_details)
        clip_model_temp_image = create_model(cfg['backbone_type'], pretrained=cfg['pretrained_weight'], design_details = design_details)
        with torch.no_grad():
            # embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            self.ZS_clip = clip_model_temp
            # Now pre-compute the frozen VL embeddings
            # all_teacher_features = []
            # # Using multiple text templates to ensure textual diversity during training
            # for single_template in IMAGENET_TEMPLATES:
            #     x = [single_template.replace("{}", name) for name in classnames]
            #     x_tokenized = torch.cat([clip.tokenize(p) for p in x])
            #     text_features = clip_model_temp.encode_text(x_tokenized.cuda())
            #     all_teacher_features.append(text_features.unsqueeze(1))

        # self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        self.fixed_embeddings = None
        ####################################################################################
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.increment = cfg['increment']

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

    def expand_prompts(self, classnames):
        print(f"Expanding prompts: original size of prompts {self.tokenized_prompts.shape}")
        
        # CSC
        # if self.n_cls > self.increment:
        #     ctx = torch.cat([self.frozon_ctx, self.ctx], dim=0)
        # else:
        #     ctx = self.ctx
        # self.register_buffer("frozon_ctx", ctx.clone().detach())
        # ctx_vectors = torch.empty(self.increment, self.n_ctx, self.ctx_dim, dtype=self.dtype)
        # nn.init.normal_(ctx_vectors, std=0.02)
        # self.ctx = nn.Parameter(ctx_vectors)
        # print(f"frozen context vector: {self.frozon_ctx.shape}")
        # print(f"context vector: {self.ctx.shape}")
        
        prompt_prefix = self.ctx_init
        
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        # self.ctx = nn.Parameter(ctx_vectors)

        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        tokenized_prompts = torch.cat([_tokenizer(p) for p in prompts]).cuda()  # (n_cls, n_tkn)

        print("copying original prompts to new one ...")
        p_size = self.tokenized_prompts.shape
        tokenized_prompts[:p_size[0], :p_size[1]] = self.tokenized_prompts
        # print(embedding.shape)
        

        with torch.no_grad():
            embedding = self.ZS_clip.token_embedding(tokenized_prompts).type(self.dtype)
            # self.ZS_image_encoder = clip_model_temp_image.visual
            # self.ZS_clip = clip_model_temp.eval()
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                # x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                x_tokenized = torch.cat([_tokenizer(p) for p in x])
                text_features = self.ZS_clip.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        print(f"fixed embeddings {self.fixed_embeddings.shape}")
        ####################################################################################
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS
        print(f"after expanding, token_prefix.shape = {self.token_prefix.shape}, token_suffix.shape = {self.token_suffix.shape}")
        
        self.n_cls = len(classnames)
        self.tokenized_prompts = tokenized_prompts
        print(f"Done expanding prompts: later size of prompts {self.tokenized_prompts.shape}")
        

    def forward(self):

        if len(self.ctx.shape) == 2:
            ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            
        # CSC
        # if self.n_cls > self.increment:
        #     ctx = torch.cat([self.frozon_ctx, self.ctx], dim=0)
        # else:
        #     ctx = self.ctx

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.transformer.get_cast_dtype()
        # self.total_epochs = cfg.epochs
        self.n_cls = len(classnames)
        self.feature_dim = clip_model.feature_dim
        
    def expand_prompts(self, classnames):
        self.n_cls = len(classnames)
        self.prompt_learner.expand_prompts(classnames)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                # zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.cuda().t()

            return logits, text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits
        else:
            return logits, logits.t()
        # return logits, logits.t()
        

def get_promptsrc(cfg, classnames):
    print(f"Loading CLIP (backbone: {cfg['backbone_type']}.{cfg['pretrained_weight']})")
    # clip_model = load_clip_to_cpu(cfg)
    design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg['prompt_depth_vision'],
                          "language_depth": cfg['prompt_depth_text'],
                          "vision_ctx": cfg['n_ctx_vision'],
                          "language_ctx": cfg['n_ctx_text']}
    
    clip_model, train_trfm, test_trfm = create_model_and_transforms(cfg['backbone_type'], pretrained=cfg['pretrained_weight'], design_details=design_details)
    # clip_model = load_clip_to_cpu(cfg).float()
    
    print("Building custom CLIP")
    model = CustomCLIP(cfg, classnames, clip_model.eval())

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
    
    return model, train_trfm, test_trfm