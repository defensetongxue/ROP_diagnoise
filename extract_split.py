import os,json
def extract_split(data_path):
    split_annote={
        'train':[],
        'val':[],
        'test':[]
    }
    os.makedirs(os.path.join(data_path,'split'),exist_ok=True)
    annotation={}
    for split in ['train','val','test']:
        with open(os.path.join(data_path,'annotations',f'{split}.json'),'r') as f:
            data_list=json.load(f)
        for data in data_list:
            annotation[data['image_name']]={
                'image_path':data['image_path'],
                'id':data['image_name'].split('.')[0],
                'Stage':data['class'],
                'Zone':0,
                'Plus':0,
            }
            split_annote[split].append(data['image_name'])
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        json.dump(annotation,f)
    with open(os.path.join(data_path,'split','0.json'),'r') as f:
        json.dump(split_annote,f)

if __name__ == '__main__':
    from config import get_config
    args=get_config()
    extract_split(args.path_tar)