import os
from .eye_quality_processer import EyeQualityProcesser
from PIL import Image
def generate_eyeQuality_result(data_path='./data'):
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
            └─────new: eye_quailty
                │
                └───new: 001.txt
                └───new: 002.txt
                └───new: ...

    This function will generate the eye quality label for
    each image in data/image in the format "<x>" in each txt file:
    This Module based on the project https://github.com/hzfu/EyeQ
    the label format is a number with meaning:
    0: Good
    1: Usable
    2: Reject
    Thanks a lot
    '''
    # Create save dir 
    save_dir=os.path.join(data_path,'eye_quality')
    os.makedirs(save_dir,exist_ok=True)

    # Init processer
    processer=EyeQualityProcesser()

    # Image list
    img_dir=os.path.join(data_path,'images')
    img_list=os.listdir(img_dir)
    label_cnt={'1':0,'2':0,'0':0}
    for image_name in img_list:
        img=Image.open(os.path.join(img_dir,image_name))
        label=processer(img,save_path=os.path.join(save_dir,f"{image_name.split('.')[0]}.txt"))
        label_cnt[str(label)]+=1
    print(label_cnt)
    
    
        