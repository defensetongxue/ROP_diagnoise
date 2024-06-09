# for each image generate a mask for edge region by analysing the pixel value
import os,json
from PIL import Image
import numpy as np
data_path='../autodl-tmp/dataset_ROP'
with open(os.path.join(data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
os.makedirs(os.path.join(data_path,'mask'),exist_ok=True)

def generate_mask(image_path,left_cut=80, right_cut=60,save_path=None):
    img=Image.open(image_path)
    img=np.array(img)/255
    img=np.sum(img,axis=-1)
    mask=np.where(img<0.26,0,1)
    # Get the width of the image
    img_width = img.shape[1]
    # Set left and right areas to zero
    mask[:, :left_cut] = 0  # Set left area to zero
    mask[:, img_width-right_cut:] = 0  # Set right area to zero

    # Convert the mask to an image and save it
    if save_path is not None:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(save_path)
    return mask
for image_name in data_dict:
    save_mask_path=os.path.join(data_path,'mask',image_name)
    generate_mask(data_dict[image_name]['image_path'],
                  save_path=save_mask_path)
    data_dict[image_name]['mask_path']=save_mask_path
with open(os.path.join(data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)