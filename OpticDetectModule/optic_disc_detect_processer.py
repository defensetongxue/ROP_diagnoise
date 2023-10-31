# this file will create an interface for the rop_dig
from . import models
import torch
from PIL import ImageEnhance,Image
import torchvision.transforms as transforms
import math
import numpy as np
import os,json
class OpticDetProcesser():
    def __init__(self,model_path='./model_save/optic.pth',config_path='./config_file/optic_disc.json'): 
        with open(config_path,'r') as f:
            cfgs=json.load(f)
        self.model,_ = getattr(models,cfgs['model']['name'])(cfgs['model'])
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
        self.distance_map={
            0:"visible",1:"near",2:"far"
        }
    def __call__(self, image_path):
        # open the image and preprocess
        with torch.no_grad():
            img=Image.open(image_path).convert('RGB')
            ori_w,ori_h=img.size

            w_ratio,h_ratio=ori_w/self.image_resize[0], ori_h/self.image_resize[1]
            img = self.transforms(img)

            # generate predic heatmap with pretrained   model
            img = img.unsqueeze(0)  # as batch size 1
            position = self.model(img.cuda())
            # the input of the 512 is to match the  mini-size of vessel model
            score_map = position.data.cpu()
            preds = decode_preds(score_map)
            preds=preds.squeeze()
            preds=preds*np.array([w_ratio,h_ratio])
            distance_pred= "visible" if torch.max(score_map)>0.15 else "near"
            return preds,distance_pred

def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 3, 'Score maps should be 3-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), -1), 1)

    maxval = maxval.view(scores.size(0), -1)
    idx = idx.view(scores.size(0), -1) + 1

    preds = idx.repeat(1, 2).float()

    preds[:, 0] = (preds[:,  0] - 1) % scores.size(2) + 1
    preds[:, 1] = torch.floor((preds[:,  1] - 1) / scores.size(2)) + 1

    pred_mask = maxval.gt(0).repeat(1, 2).float()
    preds *= pred_mask
    return preds

def decode_preds(output):
    map_width,map_height=output.shape[-2],output.shape[-1]
    coords = get_preds(output)  # float type
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        hm = output[n]
        px = int(math.floor(coords[n][0]))
        py = int(math.floor(coords[n][1]))
        if (px > 1) and (px < map_width) and (py > 1) and (py < map_height):
            diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
            coords[n] += diff.sign() *.25
    preds = coords.clone()

    # Transform back
    return preds*4 # heatmap is 1/4 to original image

class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img