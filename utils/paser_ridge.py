import  os
import json
from .api_record import api_check,api_update
def generate_ridge(json_dict,data_path):
    print(f"begin paser ridge from {json_dict}")
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        original_annoation=json.load(f)
    file_list=sorted(os.listdir(json_dict))
    print(f"read the origianl json file from {file_list}")
    for file in file_list:
        if not file.split('.')[-1]=='json':
            print(f"unexpected file {file} in json_src")
            continue
        with open(os.path.join(json,file), 'r') as f:
            data = json.load(f)
        if int(file[0])==6:
            continue
        for json_obj in data:
            image_name,new_data=parse_json(json_obj)
            if new_data["ridge_number"]>0:     
                original_annoation[image_name]['ridge']=(new_data)
    api_update(data_path,'ridge',"Location annotation for ROP lesion, including the ROP lesion coordinate")
    print("finished")
    
def parse_json(input_data):
    annotations = input_data.get("annotations", [])
    if annotations:
        result = annotations[0].get("result", [])
    image_name=input_data["file_upload"].split('-')[-1]
    new_data = {\
        "ridge_number": 0,
        "ridge_coordinate": [],
        "other_number": 0,
        "other_coordinate": [],
        "plus_number": 0,
        "plus_coordinate": [],
        "pre_plus_number": 0,
        "pre_plus_coordinate": [],
        "vessel_abnormal_number":0,
        "vessel_abnormal_coordinate":[],
    }

    for item in result:
        if item["type"] == "keypointlabels":
            # x, y = item["value"]["x"], item["value"]["y"]
            x= item["value"]["x"]*item["original_width"]/100
            y= item["value"]["y"]*item["original_height"]/100
            label = item["value"]["keypointlabels"][0]

            if label == "Ji":
                new_data["ridge_number"] += 1
                new_data["ridge_coordinate"].append((x, y))
            elif label == "Other":
                new_data["other_number"] += 1
                new_data["other_coordinate"].append((x, y))
            elif label == "Plus":
                new_data["plus_number"] += 1
                new_data["plus_coordinate"].append((x, y))
            elif label == "Pre-plus":
                new_data["pre_plus_number"] += 1
                new_data["pre_plus_coordinate"].append((x, y))
            elif label =='Stage':
                new_data["vessel_abnormal_number"]+=1
                new_data['vessel_abnormal_coordinate'].append((x,y))

    return image_name,new_data