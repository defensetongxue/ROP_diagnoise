from .ridge_segmentation_prcesser import ridge_segmentation_processer
import os 
import json
def generate_ridge(data_path):
    splits=['train','val','test']
    os.makedirs(os.path.join(data_path,'ridge_mask'),exist_ok=True)
    os.makedirs(os.path.join(data_path,'ridge_points'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'ridge_mask')}/*")
    os.system(f"rm -rf {os.path.join(data_path,'ridge_points')}/*")
    processer=ridge_segmentation_processer('point',5,150)
    for split in splits:
        with open(os.path.join(data_path,'annotations',f'{split}.json'),'r') as f:
            data_list=json.load(f)
        annotation_res=[]
        for data in data_list:
            pos_path=os.path.join(data_path,'pos_embed',data['image_name'].split('.')[0]+'.pt')
            _,annotes=processer(data['image_path'],
                                position_path=pos_path,
                                class_=data['class'],
                                save_path=os.path.join(
                data_path,'ridge_mask',data['image_name']))
            annotation_res.append(annotes)
        with open(os.path.join(data_path,'ridge_points',f'{split}.json'),'w') as f:
            json.dump(annotation_res,f)
    
        