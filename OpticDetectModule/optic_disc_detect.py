import os
from .optic_disc_detect_processer import OpticDetProcesser
from PIL import Image
def generate_OpticDetect_result(data_path='./data'):
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
            └─────new: optic_disc
                │
                └───new: 001.txt
                └───new: 002.txt
                └───new: ...

    This function will generate the optic disc coordinates for
    each image in data/image in the format "<x> <y>" in each txt file:
    if there is no optic disc, the txt file is empty
    Model training process is in https://github.com/defensetongxue/optic_disc_detection-HRNetv2-
    Algorithm is from https://github.com/HRNet/HRNet-Facial-Landmark-Detection This repository 
    is based on the facial landmark detect task, however, as the visualization, I think the result
    is acceptable and the design of the backbone is good enough
    Thanks a lot
    '''
    # Create save dir 
    save_dir=os.path.join(data_path,'optic_disc')
    os.makedirs(save_dir,exist_ok=True)

    # Init processer
    processer=OpticDetProcesser()

    # Image list
    img_dir=os.path.join(data_path,'images')
    img_list=os.listdir(img_dir)

    for image_name in img_list:
        img=Image.open(os.path.join(img_dir,image_name))
        processer(img,save_path=os.path.join(save_dir,f"{image_name.split('.')[0]}.txt"))
    
    
        