import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from .models import FR_UNet
import cv2
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
        # print(scores.shape,xmin,xmax,ymin,ymax)
        scores[ xmin:xmax,ymin:ymax] = -9
        # print(scores[ xmin:xmax,ymin:ymax])
        # raise
    maxvals_list=np.array(maxvals_list,dtype=np.float32)
    preds_list=np.array(preds_list,dtype=np.float32)
    return maxvals_list, preds_list

class ridge_segmentation_processer():
    def __init__(self,mode,point_number,point_dis=50):
        self.mode=mode
        assert mode in ['mask','point']

        if mode=='point':
            self.point_number=point_number
            self.point_dis=point_dis
        self.model=FR_UNet().cuda()
        self.model.load_state_dict(
                torch.load('./ridgeSegModule/checkpoint/ridge_seg.pth'))
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])

    def __call__(self,img_path,position_path,class_,save_path=None):
        with torch.no_grad():
            img=Image.open(img_path)
            # decomposed_images = decompose_image_into_tensors(self.img_transforms(img))
            # composed_mask = []
            # for decomposed_image in decomposed_images:
            #     decomposed_image = decomposed_image.unsqueeze(0).cuda() 
            #     mask_part = self.model(decomposed_image).detach().cpu()
            #     mask_part = torch.sigmoid(mask_part)
            #     composed_mask.append(mask_part)

            # mask = compose_tensors_into_image(composed_mask)
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
            assert save_path is not None
            if self.mode=='point':
                maxvals_list, preds_list=k_max_values_and_indices(
                    mask,self.point_number,self.point_dis)
                maxvals_list=maxvals_list.tolist()
                preds_list=preds_list.tolist()
                data={
                    'image_name':os.path.basename(img_path),
                    'image_path':img_path,
                    'point_number':self.point_number,
                    'heatmap_path':save_path,
                    'class':class_,
                    'ridge':[{
                        'coordinate':preds_list[i],
                        'value':maxvals_list[i]
                    } for i in range(len(maxvals_list))]
                }
                return mask, data
            return mask
