import pdb
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import clip
from cleanfid.fid import compute_fid
import kornia
import torch.nn.functional as F


def img_preprocess_clip(img_np):
    x = Image.fromarray(img_np.astype(np.uint8)).convert("RGB")
    T = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
    ])
    return np.asarray(T(x)).clip(0, 255).astype(np.uint8)

def img_preprocess_clip_tensor(img_t):

    T = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return T(img_t)


class CLIP_fx(nn.Module):
    def __init__(self, name="ViT-B/32", device="cuda"):
        super(CLIP_fx,self).__init__()
        self.model, _ = clip.load(name, device=device)
        self.mean = nn.parameter.Parameter(torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))#.to(device)
        self.std = nn.parameter.Parameter(torch.tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))#.to(device)
        
    
    def __call__(self, img_t):
        #print(img_t.shape)
        #print(img_t.mean())
        img_t = img_t.type(torch.float32)
        img_t = F.interpolate(img_t, size=(224,224), mode='bicubic')
        img_t = (img_t - self.mean)/self.std
       
        
        z = self.model.encode_image(img_t)
        return z

        