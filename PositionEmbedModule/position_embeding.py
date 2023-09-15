import os,json
from .pos_embed_processer import PosEmbedProcesser
from .api_record import api_check,api_update
def generate_pos_embed(data_path='./data',model_dict="./ROP_diagnoise/model_save",split_name="mini"):
    print("begin to generate position embeding ")
    # Create save dir 
    api_check(data_path,'vessel_path')
    save_dir=os.path.join(data_path,'pos_embed')
    os.makedirs(save_dir,exist_ok=True)
    os.system(f'rm -rf {save_dir}/*')
    # Init processer
    processer=PosEmbedProcesser( split_name=split_name,
                                config_path='./config_file/pos_embed.json',
                                model_dict=model_dict)

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
    
        