import os,json
def api_init(data_path,content={}):
    with open(os.path.join(data_path,'api.json'),'w') as f:
        json.dump(content,f)

def api_check(data_path,key):
    with open(os.path.join(data_path,'api.json'),'r') as f:
        api=json.load(f)
    if key in api:
        return 
    raise ValueError(f"{key} do not exit, you should get necessary intermediate data")

def api_update(data_path,key,description):
    with open(os.path.join(data_path,'api.json'),'r') as f:
        api=json.load(f)
    api[key]=description
    with open(os.path.join(data_path,'api.json'),'w') as f:
        json.dump(api,f)

def api_updates(data_path,update_dict):
    with open(os.path.join(data_path,'api.json'),'r') as f:
        api=json.load(f)
    api.update(update_dict)
    with open(os.path.join(data_path,'api.json'),'w') as f:
        json.dump(api,f)
