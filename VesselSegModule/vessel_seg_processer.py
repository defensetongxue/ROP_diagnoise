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

class VesselSegProcesser():
    def __init__(self, 
                 model_dict="./ROP_diagnoise/model_save"):
        checkpoint = torch.load(
            os.path.join(model_dict,'vessel_seg.pth'))
        loaded_state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace("module.", ""): v for k, v in loaded_state_dict.items()}
        self.model = FR_UNet().cuda()
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        # generate mask
        mask = Image.open('./VesselSegModule/mask.png')
        mask=transforms.Resize((300,400))(mask)
        mask = transforms.ToTensor()(mask)[0]
        self.mask = mask

        self.transforms = transforms.Compose([
            ContrastEnhancement(),
            transforms.Resize((300,400)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.3968], [0.1980])
            # the mean and std is cal by 12 rop1 samples
            # TODO using more precise score
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
            if save_path:
                cv2.imwrite(save_path,
                    np.uint8(predict.numpy()*255))
            return predict.numpy()
