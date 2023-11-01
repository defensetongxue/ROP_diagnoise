import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os,json

class GammaTransform:
    def __init__(self, gamma_red=0.7, gamma_green=0.9):
        self.gamma_red = gamma_red
        self.gamma_green = gamma_green
        
    def __call__(self, image):
        red_channel = image[0, :, :] ** self.gamma_red
        green_channel = image[1, :, :] ** self.gamma_green
        transformed_image = torch.stack([red_channel, green_channel, image[2, :, :]])
        return transformed_image


class BilateralFilter:
    def __init__(self, d=10, sigma_color=40, sigma_space=30):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, image):
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        filtered_image = cv2.bilateralFilter(image_np, self.d, self.sigma_color, self.sigma_space)
        filtered_tensor = torch.from_numpy(filtered_image).permute(2, 0, 1).float() / 255.0
        return filtered_tensor


class ClaheEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        red_channel = image[0, :, :]
        green_channel = image[1, :, :]
        red_eq = clahe.apply((red_channel.cpu().numpy() * 255).astype(np.uint8))
        green_eq = clahe.apply((green_channel.cpu().numpy() * 255).astype(np.uint8))
        red_tensor = torch.from_numpy(red_eq).float() / 255.0
        green_tensor = torch.from_numpy(green_eq).float() / 255.0
        enhanced_image = torch.stack([red_tensor, green_tensor, image[2, :, :]])
        return enhanced_image
if __name__=='__main__':
    from config import get_config
    args=get_config()
    # Composing the enhancers
    enhance_transform = transforms.Compose([
        GammaTransform(),
        BilateralFilter(),
        ClaheEnhancer()
    ])  
    with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    os.makedirs(os.path.join(args.data_path,'enhanced_image'),exist_ok=True)
    for image_name in data_dict:
        data=data_dict[image_name]
        image_path = data['image_path']
        image = Image.open(image_path)
        image_tensor = transforms.ToTensor()(image)
        enhanced_tensor = enhance_transform(image_tensor)
        enhanced_path=os.path.join(args.data_path,'enhanced_image',image_name)
        # Save the enhanced image
        enhanced_image_np = (enhanced_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        enhanced_image = Image.fromarray(enhanced_image_np)
        enhanced_image.save(enhanced_path)

        data_dict[image_name]['enhanced_path']=enhanced_path
    with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
        json.dump(data_dict,f)