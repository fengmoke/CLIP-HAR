import os.path as osp
import numpy as np
import random
import torch
import sys
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch import optim
from tqdm import tqdm
from PIL import Image
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import *
from prompters import *
# from clip.CODER import *


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

FEATURES_DIM = 512
CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
    'UCI':'an action of {}',
    'PAMAP2':'an action of {}',
    'MotionSense':'an action of {}',
    'WISDM':'an action of {}',
    'HHAR':'an action of {}'
}

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class TextEncoder0(nn.Module):

    def __init__(self, dataset, classnames, clip_model, res=True):
        super().__init__()
        self.res = res
        self.dataset = dataset
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.adapter_text = Adapter(FEATURES_DIM, 4).to(clip_model.dtype)
        self.eta = nn.Parameter(torch.tensor(0.6))
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.dataset]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        if self.res:
            x = self.adapter_text(text_features)
            text_features = self.eta * x + (1 - self.eta) * text_features
            text_features = x + text_features
        return text_features





class CustomCLIP5(nn.Module):

    def __init__(self, dataset, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder0(dataset, classnames, clip_model, res=False)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(512, 4).to(clip_model.dtype)
        self.ratio = nn.Parameter(torch.tensor(0.2))

            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features


class CustomCLIP0(nn.Module):

    def __init__(self, dataset, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder0(dataset, classnames, clip_model, res=False)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(FEATURES_DIM, 4).to(clip_model.dtype)
        # self.ratio = nn.Parameter(torch.tensor(0.45))
        self.ratio = 1

            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        image_features = self.ratio * x + (1 - self.ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        # print(self.ratio)

        return logits, image_features, text_features


class CustomCLIP2(nn.Module):

    def __init__(self, dataset, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder0(dataset, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter_image = Adapter(FEATURES_DIM, 4).to(clip_model.dtype)
        self.ratio = nn.Parameter(torch.tensor(0.2))

            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter_image(image_features)

        image_features = self.ratio * x + (1 - self.ratio) * image_features
        # image_features = x + image_features

        text_features = self.text_encoder()



        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = image_features @ text_features.t() / 0.07

        return logits, image_features, text_features



class CustomCLIP1(nn.Module):

    def __init__(self, dataset, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder0(dataset, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.ratio = nn.Parameter(torch.tensor(0.2))

            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))


        text_features = self.text_encoder()



        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = image_features @ text_features.t() / 0.07

        return logits, image_features, text_features

class CustomCLIP7(nn.Module):

    def __init__(self, dataset, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.linear = nn.Linear(512, 6)

            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        logits = self.linear(image_features)

        return logits



###VPT#####################
class TextEncoder1(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

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


class FixedEmbeddings():
    def __init__(self, cfg, classnames, clip_model):
        clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        prompt_prefix = "an action of"
        # print('Vision Prompting Design')
        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.VPT.N_CTX_VISION}")
        # print(f"Using fixed hand crated prompts")

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_prompts)

        self.fixed_embeddings = text_features

    def return_fixed_embeddings(self):
        return self.fixed_embeddings


class CustomCLIP3(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.embeddings = FixedEmbeddings(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder1(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None, training=False):
        logit_scale = self.logit_scale.exp()

        text_features = self.embeddings.return_fixed_embeddings().cuda()
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if training:
            return F.cross_entropy(logits, label)

        return logits, image_features, text_features

def build_custom_clip(dataset, classname, clip_model_path, adapter_num):
    clip_model, preprocess = load(clip_model_path)
    clip_model.float()
    # for param in clip_model.parameters():
    #     param.requires_grad = False
    if adapter_num == 0:
        for param in clip_model.parameters():
            param.requires_grad = False
        model = model = CustomCLIP0(dataset, classname, clip_model)
    elif adapter_num == 1:
        # for param in clip_model.parameters():
        #     param.requires_grad = False
        clip_model.eval()
        model = CustomCLIP1(dataset, classname, clip_model)
    elif adapter_num == 2:
        for param in clip_model.parameters():
            param.requires_grad = False
        model = CustomCLIP2(dataset, classname, clip_model)
    elif adapter_num == 3:
        model = CustomCLIP3(dataset, classname, clip_model)
    elif adapter_num == 4:# 微调整个视觉编码器
        for name, param in clip_model.named_parameters():
            if "text" in name:
                param.requires_grad = False
        model = CustomCLIP5(dataset, classname, clip_model)
    elif adapter_num == 5:# 微调整个CLIP
        model = CustomCLIP5(dataset, classname, clip_model)
    elif adapter_num == 6:# 微调整个文本编码器
        for name, param in clip_model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
        model = CustomCLIP5(dataset, classname, clip_model)
    # elif adapter_num == 7: # 线性探测
    #     for param in clip_model.parameters():
    #         param.requires_grad = False
    #     model = CustomCLIP7(dataset, classname, clip_model)
    elif adapter_num == 8: # 视觉提示-padding
        for param in clip_model.parameters():
            param.requires_grad = False
        model = CustomCLIP5(dataset, classname, clip_model)
        visual_prompt = padding()
        return model, preprocess, visual_prompt
    elif adapter_num == 9: # 视觉提示-padding
        for param in clip_model.parameters():
            param.requires_grad = False
        model = CustomCLIP5(dataset, classname, clip_model)
        visual_prompt = fixed_patch()
        return model, preprocess, visual_prompt
    elif adapter_num == 10: # 视觉提示-random_patch
        for param in clip_model.parameters():
            param.requires_grad = False
        model = CustomCLIP5(dataset, classname, clip_model)
        visual_prompt = random_patch()
        return model, preprocess, visual_prompt
    return model, preprocess


