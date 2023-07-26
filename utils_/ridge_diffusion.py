import json
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
from scipy.ndimage import zoom
import os
from scipy.ndimage import gaussian_filter
from .api_record import api_check,api_update
def generate_ridge_diffusion(data_path):
    print("begin to generate ridge annotations")
    api_check(data_path,'ridge')
    os.makedirs(os.path.join(data_path,'ridge_diffusion'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'ridge_diffusion')}/*")
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    for image_name in data_list:
        data=data_list[image_name]
        mask = generate_diffusion_heatmap(data['image_path']
                                          ,data['ridge']['ridge_coordinate'], 
                                          factor=0.5,
                                          Gauss=False)
        mask_save_name=data['id']+'.png'
        mask_save_path=os.path.join(data_path,'ridge_diffusion',mask_save_name)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_save_path)
        data_list[image_name]['ridge_diffusion_path']=mask_save_path
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_list,f)
    api_update(data_path,'ridge_diffusion_path','path to ridge diffusion mask')
    print("finish")
def generate_diffusion_heatmap(image_path,points,factor=0.5,Gauss=False):
    '''
    factor is the compress rate: for stability, we firt apply a conv layer to 
    downsample the original data
    '''
    img_tensor=imge_tensor_compress(image_path,factor)

    img = Image.open(image_path).convert('RGB')
    img=transforms.ToTensor()(img)
    heatmap_width=int(img.shape[1]*factor)
    heatmap_height=int(img.shape[2]*factor)
    heatmap = np.zeros((heatmap_width, heatmap_height), dtype=np.float32)
    new_points=[]
    for x,y in points:
        new_points.append([int(x*factor),int(y*factor)])
    points=np.array(new_points)

    heatmap = generate_heatmap(heatmap,img_tensor,points)
    if Gauss:
        Gauss_heatmap=gaussian_filter(heatmap,3)
        heatmap=np.where(heatmap>Gauss_heatmap,heatmap,Gauss_heatmap)
    mask=heatmap2mask(heatmap,int(1/factor))
    return mask

def heatmap2mask(heatmap,factor=4):
    mask = zoom(heatmap, factor)
    return mask

def norm_tensor(t):
    min_v=t.min()
    r_val=t.max()-min_v
    return (t-min_v)/r_val

def visual_mask(image_path, mask,save_path='./tmp.jpg'):
    # Open the image file.
    image = Image.open(image_path).convert("RGBA")  # Convert image to RGBA

    # Create a blue mask.
    mask_np = np.array(mask)
    mask_blue = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)  # 4 for RGBA
    mask_blue[..., 2] = 255  # Set blue channel to maximum
    mask_blue[..., 3] = (mask_np * 127.5).astype(np.uint8)  # Adjust alpha channel according to the mask value

    # Convert mask to an image.
    mask_image = Image.fromarray(mask_blue)

    # Overlay the mask onto the original image.
    composite = Image.alpha_composite(image, mask_image)

    # Convert back to RGB mode (no transparency).
    rgb_image = composite.convert("RGB")

    # Save the image with mask to the specified path.
    rgb_image.save(save_path)

def imge_tensor_compress(img_path,factor):
    img=Image.open(img_path)
    img=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])(img).unsqueeze(0)

    # Calculate patch size based on the downscale factor
    patch_size = int(1/factor)
    
    # Create averaging kernel
    kernel = torch.full((1, img.shape[1], patch_size, patch_size), 1./(patch_size * patch_size))
    
    # Apply convolution operation to get patch embedding
    img =torch.nn.functional.conv2d(img, kernel, stride=patch_size)
    img=img.squeeze()
    return img

def generate_heatmap(heatmap,img_tensor, points):
    points=generate_point_sequence(points)
    for i in range(points.shape[0]-1):
        heatmap=diffusion(heatmap,img_tensor,points[i],points[i+1])

    return heatmap

def generate_point_sequence(points):
    points = points[points[:,0].argsort()] if (max(points[:,0]) - min(points[:,0])) > (max(points[:,1]) - min(points[:,1])) else points[points[:,1].argsort()]
    return points

def get_distance(p0, p1):
    return ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)**0.5
def get_similarity(img_tensor,p0,p1):
    return 1-((img_tensor[p0[1],p0[0]]-img_tensor[p1[1],p1[0]])**2)


def diffusion(heatmap,img_tensor,p0,p1):
    heatmap[p0[1],p0[0]]=1
    heatmap[p1[1],p1[0]]=1
    px=p0
    now_distance=get_distance(px,p1)
    while(now_distance>1):
        max_simi=-9e5
        rem=[]
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                p=[px[0]+i,px[1]+j]
                if heatmap[p[1],p[0]]>0:
                    continue
                if get_distance(p,p1)>now_distance:
                    continue
                simi=get_similarity(img_tensor,p,p0)+get_similarity(img_tensor,p,p1)

                if simi>max_simi:
                    max_simi=simi
                    rem=[i,j]
        if len(rem)==0:
            return heatmap
        px=[px[0]+rem[0],px[1]+rem[1]]
        now_distance=get_distance(px,p1)
        heatmap[px[1],px[0]]=1
    return heatmap
