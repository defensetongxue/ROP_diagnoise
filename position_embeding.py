import os
from .pos_embed_processer import PosEmbedProcesser
from PIL import Image
def generate_posEmbed_result(data_path='./data'):
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
    # Create save dir 
    save_dir=os.path.join(data_path,'pos_embed')
    os.makedirs(save_dir,exist_ok=True)
    os.system(f'rm -rf {save_dir}/*')
    # Init processer
    processer=PosEmbedProcesser(model_name='ViT',
                                vessel_resize=256,
                                image_orignal_size=(1200,1600),#todo
                                patch_size=32)

    # Image list
    img_dir=os.path.join(data_path,'vessel_seg')
    img_list=os.listdir(img_dir)
    
    for image_name in img_list:
        img=Image.open(os.path.join(img_dir,image_name)).convert('RGB')
        processer(img,save_path=os.path.join(save_dir,image_name.split('.')[0]+'.pt'))

    
        