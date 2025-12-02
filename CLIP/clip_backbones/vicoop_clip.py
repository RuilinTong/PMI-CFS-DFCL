import torch
import torch.nn as nn


# from clip import clip
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import get_tokenizer
# from open_clip.tokenizer import _tokenizer, tokenize
from .imagenet_templates import IMAGENET_TEMPLATES
from .myclip import create_model, create_model_and_transforms


# _tokenizer = get_tokenizer('ViT-B-16')


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.encode_text
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.transformer.get_cast_dtype()
        # self.total_epochs = cfg.epochs
        self.n_cls = len(classnames)
        self.feature_dim = clip_model.feature_dim

    def encode_text(self, text):
        return self.text_encoder(text)

    def encode_image(self, image):
        return self.image_encoder(image)

    def forward(self, image, text=None, label=None, normalize=True):
        # tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # Compute the prompted image and text features
        if text is not None:
            text_features = self.text_encoder(text)
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = None
        image_features, _ = self.image_encoder(image.type(self.dtype))
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        # logits = logit_scale * image_features @ text_features.t()
        
        # return logits, logits.t()
        return image_features, text_features, self.logit_scale

        # return logits, logits.t()
        

def get_vicoop(cfg, classnames):
    print(f"Loading CLIP (backbone: {cfg['backbone_type']}.{cfg['pretrained_weight']})")
    # clip_model = load_clip_to_cpu(cfg)
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg['prompt_depth_vision'],
                      "language_depth": 0,
                      "vision_ctx": cfg['n_ctx_vision'],
                      "language_ctx": 0}
    
    clip_model, train_trfm, test_trfm, tokenizer = create_model_and_transforms(cfg['backbone_type'], pretrained=cfg['pretrained_weight'], design_details=design_details)
    # clip_model = load_clip_to_cpu(cfg).float()
    
    print("Building custom CLIP")
    model = CustomCLIP(cfg, classnames, clip_model.cuda().eval())

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
    
    return model, train_trfm, test_trfm, tokenizer
