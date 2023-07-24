import os,json
def modify_ridge(data_path):
    for split in ['train','val','test']:
        with open(os.path.join(data_path,'ridge',f"{split}.json"),'r') as f:
            ridge_data=json.load(f)
        ridge_dict={i['image_name']:i["ridge_number"] for i in ridge_data}
        with open(os.path.join(data_path,'annotations',f"{split}.json"),'r') as f:
            data_list=json.load(f)
        new_annotation=[]
        for data in data_list:
            if data['class']>0 and ridge_dict[data['image_name']]<=0:
                data['class']=0
            new_annotation.append(data)
        with open(os.path.join(data_path,'annotations',f"{split}.json"),'w') as f:
            json.dump(f,new_annotation)
if __name__=='__main__':
    from config import get_config
    args=get_config()
    modify_ridge(args.path_tar)