# this file will create an interface for the rop_dig
from .models import dense121_mcs
import torch
from PIL import Image, ImageCms
import torchvision.transforms as transforms
import numpy as np
import cv2

class EyeQualityProcesser():
    def __init__(self):
        self.model = dense121_mcs(n_class=3)
        checkpoint = torch.load(
            './EyeQualityModule/checkpoint/DenseNet121_v3_v1.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()
        self.model.eval()

        self.transform1 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        ])

        self.transform2=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")

        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")



    def __call__(self, img,save_path=None):
        # open the image and preprocess
        # img = Image.open(img_path)
        if img.size[0]>3: 
            img=img.convert('RGB')
        img = self.transform1(img)
        img_hsv = img.convert("HSV")
        img_lab = ImageCms.applyTransform(img, self.rgb2lab_transform)

        img_rgb = np.asarray(img).astype('float32')
        img_hsv = np.asarray(img_hsv).astype('float32')
        img_lab = np.asarray(img_lab).astype('float32')

        if self.transform2 is not None:
            img_rgb = self.transform2(img_rgb)
            img_hsv = self.transform2(img_hsv)
            img_lab = self.transform2(img_lab)

        # generate predic
        img_rgb = img_rgb.unsqueeze(0).cuda()
        img_hsv = img_hsv.unsqueeze(0).cuda()
        img_lab = img_lab.unsqueeze(0).cuda()
        _, _, _, _,pre = self.model(img_rgb,img_hsv,img_lab)

        pred=torch.max(pre[0].cpu(),dim=0)[1]
        pred=int(pred)
        if save_path:
            with open(save_path,'w') as f:
                f.write(str(pred))
        return pred
