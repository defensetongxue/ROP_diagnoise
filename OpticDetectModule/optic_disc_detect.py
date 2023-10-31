import os,json
from .optic_disc_detect_processer import OpticDetProcesser
from .optic_disc_cls_processer import OpticClsProcesser
from PIL import Image
import numpy as np
def generate_optic_disc_location(data_path='./data',
                                 split_name='mini',
                                 model_dict="./ROP_diagnoise/model_save"):
    '''
    '''

    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    processer=OpticDetProcesser(model_path=os.path.join(model_dict,f'v_{split_name}_optic_disc.pth'),
            config_path='./config_file/optic_disc.json'
    )
    processer_u=OpticDetProcesser(model_path=os.path.join(model_dict,f'u_{split_name}_optic_disc.pth'),
            config_path='./config_file/optic_disc.json'
    )
    class_model=OpticClsProcesser(model_path=os.path.join(model_dict,f'{split_name}_optic_disc_cls.pth'),
                                  config_path='./config_file/optic_disc_cls.json')
    mask=Image.open('./OpticDetectModule/mask.png').convert('L')
    mask=np.array(mask)
    mask[mask>0]=1
    new={}
    cnt=0
    for image_name in data_dict:
        image_name="5168.jpg"
        cnt+=1
        data=data_dict[image_name]
        new[image_name]={}
        coordinate,distance=processer(data['image_path'])
        if distance=='visible':
            coordinate=coordinate.tolist()
            coordinate=[int(i) for i in coordinate]
            new[image_name]['optic_disc_pred']={
            'position':coordinate,
            'distance':distance
            }
            continue
        coordinate,_=processer_u(data['image_path'])
        coordinate=coordinate.numpy()
        point=[int(coordinate[0]),int(coordinate[1])]
        x,y=find_nearest_zero(mask,point)
        distance=class_model(data['image_path'])
        new[image_name]['optic_disc_pred']={
            'position':[x,y],
            'distance':distance}

    with open(os.path.join(data_path,'optic_dict.json'),'w') as f:
        json.dump(new,f)
    

def find_nearest_zero(mask, point):
    h, w = mask.shape
    cx, cy = w // 2, h // 2  # center of the image
    x, y = point

    if x>=1600 or y>=1200 or mask[y, x] == 0 :
        return point

    # Get the direction vector for stepping along the line
    direction = np.array([x - cx, y - cy])
    direction = direction / np.linalg.norm(direction)

    # Step along the line until we hit a zero in the mask
    while 0 <= x < w and 0 <= y < h:
        if  y>=1200:
            return int(x),int(y-1)
        if x>=1600:
            return int(x-1),int(y) 
        if  mask[int(y), int(x)] == 0:
            return int(x), int(y)
        
        x += direction[0]
        y += direction[1]

    return None
