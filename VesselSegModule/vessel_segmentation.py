import os
from .vessel_seg_processer import VesselSegProcesser
from PIL import Image
def generate_vessel_result(data_path='./data'):
    '''
    This funtion should be exited after the data cleasning. 
    └───data
            │
            └───images
            │   │
            │   └───001.jpg
            │   └───002.jpg
            │   └───...
            │
            └───annotations
            |   │
            |   └───train.json
            |   └───valid.json
            |   └───test.json
            └─────new: vessel_seg
                │
                └───new: 001.jpg
                └───new: 002.jpg
                └───new: ...

    This function will generate the blood vessel segmentation result for
    each image in data/image
    Model training process is in https://github.com/defensetongxue/Vessel_segmentation
    most of the code in the repository above in from https://github.com/lseventeen/FR-UNet
    Thanks a lot
    '''
    # Create save dir 
    save_dir=os.path.join(data_path,'vessel_seg')
    os.makedirs(save_dir,exist_ok=True)

    # Init processer
    processer=VesselSegProcesser(model_name='FR_UNet')

    # Image list
    img_dir=os.path.join(data_path,'images')
    img_list=os.listdir(img_dir)

    for image_name in img_list:
        img=Image.open(os.path.join(img_dir,image_name))
        processer(img,save_path=os.path.join(save_dir,image_name))

    
        