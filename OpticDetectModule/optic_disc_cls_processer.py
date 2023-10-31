# this file will create an interface for the rop_dig
from .models import cls_models
import torch
from PIL import ImageEnhance,Image
import torchvision.transforms as transforms
import math
import numpy as np
import os,json
class OpticClsProcesser():
    def __init__(self,model_path='./model_save/optic.pth',config_path='./config_file/optic_disc_cls.json'): 
        with open(config_path,'r') as f:
            cfgs=json.load(f)
        self.model,_ = cls_models(cfgs['model']['name'],cfgs['model'])
        self.model.load_state_dict(
            torch.load(model_path))
        self.model.cuda()
        self.model.eval()
        self.image_resize=cfgs['image_resize']
        # Transform define
        self.transforms = transforms.Compose([
            ContrastEnhancement(),
            transforms.Resize(cfgs['image_resize']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])
        ])
        self.distance_map={0:"near",1:"far"
        }
    def __call__(self, image_path):
        # open the image and preprocess
        with torch.no_grad():
            img=Image.open(image_path).convert('RGB')
            img = self.transforms(img)

            # generate predic heatmap with pretrained   model
            img = img.unsqueeze(0)  # as batch size 1
            preds = self.model(img.cuda()).cpu()
            distance_pred= torch.argmax(preds.squeeze())
            return self.distance_map(distance_pred)

class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img

