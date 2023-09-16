import torch
import numpy as np
import os,json
from torchvision import transforms
from PIL import Image,ImageEnhance
import torch.nn.functional as F
from .models import FR_UNet
import cv2

class ridge_segmentation_processer():
    def __init__(self,point_number,point_dis=50,
                 model_path="./ROP_diagnoise/model_save",
                 config_path="./config_file/ridge_seg.json"):
        with open(config_path,'r') as f:
            cfgs=json.load(f)
        self.point_number=point_number
        self.point_dis=point_dis
        self.model=FR_UNet(cfgs['model']).cuda()
        self.model.load_state_dict(
                torch.load(model_path))
        self.model.eval()
        self.img_transforms=transforms.Compose([
            ContrastEnhancement(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])

    def __call__(self,img_path,position_path,save_path=None):
        with torch.no_grad():
            img=Image.open(img_path).convert('RGB')
            img_tensor = self.img_transforms(img)
            pos_embed=torch.load(position_path)
            pos_embed=F.interpolate(pos_embed[None,None,:,:], size=img_tensor.shape[-2:], mode='nearest')
            pos_embed=pos_embed.squeeze()

            img=img_tensor.unsqueeze(0).cuda()
            pos_embed=pos_embed.unsqueeze(0).cuda()
            output_img = self.model((img,pos_embed)).cpu()
            # Resize the output to the original image size
            mask=torch.sigmoid(output_img).numpy()
            if save_path:
                cv2.imwrite(save_path,
                    np.uint8(mask*255))
            
            maxvals_list, preds_list=k_max_values_and_indices(
                mask,self.point_number,self.point_dis)
            maxvals_list=maxvals_list.tolist()
            preds_list=preds_list.tolist()
            data={
                'point_number':self.point_number,
                'heatmap_path':save_path,
                'coordinate':preds_list,
                'value':maxvals_list
            }
            return mask, data

def k_max_values_and_indices(scores, k,r=50):

    preds_list = []
    maxvals_list = []

    for _ in range(k):
        idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)

        maxval = scores[idx]

        maxvals_list.append(maxval)
        preds_list.append(idx)

        # Clear the square region around the point
        x, y = idx[0], idx[1]
        xmin, ymin = max(0, x - r // 2), max(0, y - r // 2)
        xmax, ymax = min(scores.shape[0], x + r // 2), min(scores.shape[1], y + r // 2)
        scores[ xmin:xmax,ymin:ymax] = -9
    maxvals_list=np.array(maxvals_list,dtype=np.float32)
    preds_list=np.array(preds_list,dtype=np.float32)
    return maxvals_list, preds_list

class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img