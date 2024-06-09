import json,os
fo_shan_path='../autodl-tmp/dataset_ROP'
shen_path='../autodl-tmp/shen_try'
path_list=[fo_shan_path,shen_path]

# get distribution for foshan dataset
with open(os.path.join(fo_shan_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
with open(os.path.join(fo_shan_path,'split','1.json'),'r') as f:
    full_split=json.load(f)
with open(os.path.join(fo_shan_path,'split','clr_1.json'),'r') as f:
    clr_split=json.load(f)
    
# merge all the split
foshan_name_list=[]
for split in full_split:
    for image_name in full_split[split]:
        foshan_name_list.append(image_name)

clr_foshan_list=[]
for split in clr_split:
    for image_name in clr_split[split]:
        clr_foshan_list.append(image_name)
foshan_condition={
    "s1":0,"z1":0,"s2":0,"z2":0,"s3":0,"z3":0,"no_use":0,'norm':0,'total':0
}
for image_name in foshan_name_list:
    foshan_condition['total']+=1
    if image_name in clr_foshan_list:
        data=data_dict[image_name]
        if data['stage']>0:
            foshan_condition['s'+str(data['stage'])]+=1
            if data['zone']==0:
                if 'zone_pred' in data:
                    foshan_condition['z'+str(data['zone_pred']['zone'])]+=1
                else:
                    print(image_name) 
                continue
            foshan_condition['z'+str(data['zone'])]+=1
        else:
            foshan_condition['norm']+=1
    else:
        foshan_condition["no_use"]+=1
print(foshan_condition)