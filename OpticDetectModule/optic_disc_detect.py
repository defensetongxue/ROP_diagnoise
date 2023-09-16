import os,json
from .optic_disc_detect_processer import OpticDetProcesser
from PIL import Image
def generate_optic_disc_location(data_path='./data',
                                 split_name='mini',
                                 model_dict="./ROP_diagnoise/model_save"):
    '''
    '''

    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    processer=OpticDetProcesser(model_path=os.path.join(model_dict,f'{split_name}_optic_disc.pth'),
            config_path='./config_file/optic_disc.json'
    )
    for image_name in data_dict:
        data=data_dict[image_name]
        coordinate,distance=processer(data['image_path'])
        data_dict[image_name]['optic_disc_pred']={
            'position':coordinate.tolist(),
            'distance':distance
        }
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_dict,f)
    
        