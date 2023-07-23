import json
import os
from config import get_config
def modify_class(file_path,modify_file,from_class,to_class):
    with open(file_path,'r') as f:
        data_list=json.load(f)
    with open(modify_file,'r') as f:
        modify_list=[]
        for line in f.readlines():
            modify_list.append(line.strip())
    new_annotation=[]
    for data in data_list:
        if data['image_name'] in modify_list and data['class']==from_class:
            data['class']=to_class
        new_annotation.append(data)
    with open(file_path,'w') as f:
        json.dump(new_annotation,f)
if __name__=='__main__':
    args=get_config()
    data_path=args.path_tar
    for split in ['train','val','test']:
        modify_class(
            file_path=os.path.join(data_path,'ridge',f"{split}.json"),
            modify_file='./modify_list.txt',
            from_class=3,
            to_class=2
        )

            