import os,json
from .vessel_seg_processer import VesselSegProcesser
from .api_record import api_update
def generate_vessel(data_path='./data',model_dict="./ROP_diagnoise/model_save"):
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

    
        