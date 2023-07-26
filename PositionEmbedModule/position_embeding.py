import os,json
from .pos_embed_processer import PosEmbedProcesser
from utils_ import api_check,api_update
def generate_pos_embed(data_path='./data'):
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
            └─────new: pos_embed
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
    print("begin to generate position embeding ")
    # Create save dir 
    api_check(data_path,'vessel_path')
    save_dir=os.path.join(data_path,'pos_embed')
    os.makedirs(save_dir,exist_ok=True)
    os.system(f'rm -rf {save_dir}/*')
    # Init processer
    processer=PosEmbedProcesser(model_name='ViT',
                                vessel_resize=256,
                                image_orignal_size=(1200,1600),#todo
                                patch_size=32)

    # Image list
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    for image_name in data_list:
        save_path=os.path.join(save_dir,data_list[image_name]['id']+'.pt')
        processer(data_list[image_name]['vessel_path'],
                  save_path=save_path)
        data_list[image_name]['pos_embed_path']=save_path
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_list,f)
    api_update(data_path,'pos_embed_path','path to position embeding using for ridge segmentation')
    print("finish")
    
        