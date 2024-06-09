from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import json
from random import shuffle
from shutil import copy
from config import get_config
def visual_point_annote(point_list, image_path, save_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font_path = './arial.ttf'
    font = ImageFont.truetype(font_path, size=40)

    # Sort the point list by x-coordinate
    point_list.sort(key=lambda x: x[1])

    for i, (x, y) in enumerate(point_list, start=1):
        # Draw a solid white circle at the point's coordinate
        draw.ellipse((x-15, y-15, x+15, y+15), fill="white")
        # Draw a number in the top left of the point with font size 40 and color white
        draw.text((x-30, y-50), str(i), fill="black", font=font)  # Change text color to black for visibility

    # Save the result in the save_path
    image.save(save_path)


def visual_diffusion_path(image_path, mask_path,save_path):
    # Open the image file.
    image = Image.open(image_path).convert("RGBA")  # Convert image to RGBA
    # Create a blue mask.
    mask=Image.open(mask_path).convert('L')
    mask_np = np.array(mask)/255
    mask_blue = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)  # 4 for RGBA
    mask_blue[..., 2] = 255  # Set blue channel to maximum
    mask_blue[..., 3] = (mask_np * 127.5).astype(np.uint8)  # Adjust alpha channel according to the mask value

    # Convert mask to an image.
    mask_image = Image.fromarray(mask_blue)

    # Overlay the mask onto the original image.
    composite = Image.alpha_composite(image, mask_image)
    # Define font and size.
    rgb_image = composite.convert("RGB")
    # Save the image with mask to the specified path.
    rgb_image.save(save_path)
args = get_config()  # Assuming get_config() is defined elsewhere and works as intended
visual_dir = './experiments/ridge_diffusion'
os.makedirs(visual_dir, exist_ok=True)
os.system(f'rm -rf {visual_dir}/*')

with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
    data_dict = json.load(f)

visual_number = 20
# Randomly iterate through data_dict
image_names = list(data_dict.keys())
shuffle(image_names)
point_number_list=[]
for image_name in image_names:
    data=data_dict[image_name]
    if 'ridge' not in data or 'ridge_diffusion_path' not in data:
        continue
    point_number_list.append(len(data['ridge']["ridge_coordinate"]))
    visual_number-=1
    if visual_number<0:
        continue
    
    points_list=data['ridge']["ridge_coordinate"]
    save_dir=os.path.join(visual_dir,image_name[:-4])
    os.makedirs(save_dir, exist_ok=True)
    copy(data['image_path'],os.path.join(save_dir,image_name))
    visual_diffusion_path(
        image_path=data['image_path'],
         mask_path=data['ridge_diffusion_path'],
         save_path=os.path.join(save_dir,'ridge_diffusion.jpg'))
    visual_point_annote(
        point_list=points_list,
        image_path=data['image_path'],
        save_path=os.path.join(save_dir,'annote.jpg')
    )
import numpy as np

# Assuming the rest of your script is correctly populating point_number_list

# Calculate max, min, and mean of the point_number_list
max_points = max(point_number_list)
min_points = min(point_number_list)
mean_points = np.mean(point_number_list)  # Using numpy for mean calculation

print(f"Max: {max_points}, Min: {min_points}, Mean: {mean_points:.2f}")
