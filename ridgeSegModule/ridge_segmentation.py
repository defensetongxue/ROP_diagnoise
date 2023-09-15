from .ridge_segmentation_prcesser import ridge_segmentation_processer
import os 
import json
from .api_record import api_check,api_update
def generate_ridge_segmentation(data_path,model_dict="./ROP_diagnoise/model_save"):
    api_check(data_path,'pos_embed_path')
    print("generate ridge segmentation")
    os.makedirs(os.path.join(data_path,'ridge_segmentation'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'ridge_segmentation')}/*")
    processer=ridge_segmentation_processer(5,150,model_dict)
    
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    for image_name in data_list:
        data=data_list[image_name]
        _,annotes=processer(data['image_path'],
                            position_path=data['pos_embed_path'],
                            save_path=os.path.join(
            data_path,'ridge_segmentation',data['image_name']))
        data_list[image_name]['ridge_seg']=annotes
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_list,f)
    api_update(data_path,'ridge_seg',{
                'point_number':"find the max k number as candidate ridge point",
                'heatmap_path':'ridge_segmentation mask pathh',
                'coordinate':"coordinate for candidate ridge points",
                'value':"pred value for candidate points"
            })
    print("finish")
    
        