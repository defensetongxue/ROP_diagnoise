import os,json
from .eye_quality_processer import EyeQualityProcesser
from utils_ import api_update
def generate_quality(data_path='./data'):
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
    This function will generate the eye quality label for
    each image 
    This Module based on the project https://github.com/hzfu/EyeQ
    the label format is a number with meaning:
    0: Good
    1: Usable
    2: Reject
    Thanks a lot
    '''

    # Init processer
    print("begin to generate fundus image quality")
    processer=EyeQualityProcesser()

    # Image list
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    label_cnt={'1':0,'2':0,'0':0}
    for image_name in data_list:
        label=processer(data_list[image_name]['image_path'])
        data_list[image_name]['quality']=label
        label_cnt[str(label)]+=1
    print(label_cnt)
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_list,f)
    api_update(data_path,'quality','fundus image quality 0:Good 1:Usable 2:Reject')
    print("finish")
    
        