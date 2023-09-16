import os,json
from .vessel_seg_processer import VesselSegProcesser
from ..PositionEmbedModule.api_record import api_update
def generate_vessel(data_path='./data',model_dict="./ROP_diagnoise/model_save"):
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
    print("begin to generate vessel segmentation mask")
    save_dir=os.path.join(data_path,'vessel_seg')
    os.makedirs(save_dir,exist_ok=True)

    # Init processer
    processer=VesselSegProcesser(model_dict=model_dict)

    # Image list
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    for image_name in data_list:
        save_path=os.path.join(save_dir,image_name)
        processer(data_list[image_name]['image_path'],
                  save_path=save_path)
        data_list[image_name]['vessel_path']=save_path
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_list,f)
    api_update(data_path,'vessel_path','path to vessel segmentation mask')
    print("finish")

    
        