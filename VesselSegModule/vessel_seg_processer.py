# this file will create an interface for the rop_dig
from .models import FR_UNet
import torch
from PIL import Image,ImageEnhance
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img
class NormperImage():
    def __init__(self,mask_tensor):
        self.mask_tensor=mask_tensor.unsqueeze(0)
    def __call__(self,n):
        # Mask the image tensor where mask values are greater than 0
        masked_values = n[self.mask_tensor > -1]
        # Compute the min and max values from the masked tensor
        min_val = torch.min(masked_values)
        max_val = torch.max(masked_values)

        # Normalize the image tensor using these min and max values
        normalized_n = (n - min_val) / (max_val - min_val)
        return normalized_n
    
class VesselSegProcesser():
    def __init__(self, 
                 model_dict="./ROP_diagnoise/model_save"):
        checkpoint = torch.load(
            os.path.join(model_dict,'vessel_seg.pth'))
        loaded_state_dict = checkpoint['state_dict']
        self.model = FR_UNet().cuda()
        self.model.load_state_dict(loaded_state_dict)
        self.model.eval()
        # generate mask
        mask = Image.open('./VesselSegModule/mask.png')
        mask=transforms.Resize((800,800))(mask)
        mask = transforms.ToTensor()(mask)[0]
        self.mask = mask

        self.transforms = transforms.Compose([
            ContrastEnhancement(),
            transforms.Resize((800,800)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.31269], [0.166996]),
            NormperImage(mask)
        ])

    def __call__(self, img_path,save_path=None):
        img=Image.open(img_path)
        img=self.transforms(img)
        # open the image and preprocess
        with torch.no_grad():
            mask_part = self.model(img.cuda().unsqueeze(0)).detach().cpu()
            mask_part = torch.sigmoid(mask_part.squeeze())
            print(mask_part.shape)
            # mask
            predict = torch.where(self.mask < 0.1, self.mask, mask_part)
            predict= torch.where(mask_part>0.01,torch.ones_like(mask_part),mask_part)
            if save_path:
                cv2.imwrite(save_path,
                    np.uint8(predict.numpy()*255))
            return predict.numpy()
