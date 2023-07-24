import os
from PIL import Image
import numpy as np
import shutil
import json
from config import get_config
from utils_ import generate_data_augument
class generate_data_processer():
    def __init__(self,src_path="../autodl-tmp/data_original",
                 tar_path="../autodl-tmp/dataset_ROP",
                 spilt_train=0.7,
                 spilt_val=0.2,clear=False):
        '''
        find the original data in {src_path} and 
        generate dataset in {target_patg} with the style:
    
        └───data
            │
            └───images
            │   │
            │   └───001.jpg
            │   └───002.jpg
            │   └───...
            │
            └───annotations
                │
                └───train.json
                └───valid.json
                └───test.json
        The json format is
        {
            "id": <image id> : number,
            "image_name": <image_name> : str,
            "image_name_original": <image_name original> : str,
            "image_path": <image_path> : str,
            "image_path_original": <image_path in original dictionary> : str,
            "class": <class> : number
        }

        Note-1: there is much images with strong similarity in original dataset,
        we have to ensure they not be spilt in different subset

        Note-2: The label is imbalance as the stage 3-5 ROP prety rare, 
        we have to keep the proportion of each class balance in subsets
        '''
        super(generate_data_processer,self).__init__()
        self.src_path=src_path
        self.tar_path=tar_path

        # Create essencial folder
        os.makedirs(tar_path,exist_ok=True)
        if clear:
            os.system(f"rm -rf {tar_path}/*") # clear exited data
        os.makedirs(os.path.join(tar_path,"images"),exist_ok=True)
        os.makedirs(os.path.join(tar_path,"annotations"),exist_ok=True)

        # Get datacnt and label_map
        self.data_cnt=self.get_condition()
        self.label_map={}
        for i,key in enumerate(sorted(self.data_cnt.keys())):
            self.label_map[key]=i
        self.class_number=len(self.data_cnt.keys())
        self.split_ratio={
            "train":spilt_train,
            "val":spilt_val,
            "test":1-spilt_train-spilt_val,
        }
        self.map=self.map_list('./map_list.txt')
        print(self.label_map)
    def map_list(self,map_list_file):
        result_dict = {}

        with open(map_list_file, 'r') as file:
            for line in file:
                a, b = line.strip().split(' ', 1)
                result_dict[b] = a
        return result_dict
    
    def get_map_name(self,file_path):
        if file_path in self.map:
            return self.map[file_path]
        print(file_path)
        raise

    def get_json(self,id:int,
                 image_name:str,
                 image_name_original:str,
                 image_path:str,
                 image_path_original:str,
                 label:str):
        return {
            "id": id,
            "image_name": image_name,
            "image_name_original":image_name_original,
            "image_path": image_path,
            "image_path_original": image_path_original,
            "class": self.label_map[label]
        }
    def get_label(self,file_name: str,file_dir:str,logger=True):
        '''
        task: stage the rop,
        1,2,3,4,5 as str is the stage rop
        0 no-rop
        6: 消退期
        -1 待排
        '''
        file_str=file_name.replace(" ","")

        stage_list=["1","2","3","4","5","行","退"]
        if file_str.startswith("ROP"):
            # pos_cnt=pos_cnt+1
            stage=(file_str[file_str.find("期")-1])
            if stage=='p': # no "期"，return p of .jpg
                if logger:
                    print(os.path.join(file_dir,file_name))
                return "-1"
            assert stage in stage_list,"unexpected ROP stage : {} in file {}".format(stage,file_str)
            if stage=="行" or stage=="退" :
                # return "6" # ignore the "退行期"
                return "-1"
            return stage
        else:
            return "0" 
        
    def get_condition(self):
        '''
        Iterate the original dataset but do nothing, the main contribution
        of this function is:
            1. get the number of each stage ROP to define the data spilt (Note 2)

       
        '''
        data_cnt = {}

        for person_file in os.listdir(self.src_path):
            eye_file_name = os.path.join(self.src_path, person_file)
            
            if not os.path.isdir(eye_file_name):
                continue
            for eye_file in os.listdir(eye_file_name):
                # OS/OD
                file_dic = os.path.join(self.src_path, person_file, eye_file)
                if not os.path.isdir(file_dic):
                    continue
                for file in os.listdir(file_dic):
                    # if the data can be used
                    if not file.endswith(".jpg"):
                        continue 
                    try:
                        image=Image.open(os.path.join(file_dic,file))
                    except:
                        print("{} can not open".format(
                            os.path.join(file_dic,file)))
                        continue
                    # generate vessel and saved
                    label = self.get_label(file,file_dic)

                    if label=="-1":
                        # unexpectedd stage
                        continue
                    if label in data_cnt:
                        data_cnt[label] +=1
                    else:
                        data_cnt[label] = 1
        return data_cnt
    
    def paser(self):
        '''
        Iterate the original dataset and generate data in {tar_path}
        '''
        print(f"read data from {self.src_path} and generate cleansinged data in {self.tar_path}")
        train_annotations=[]
        val_annotations=[]
        test_annotations=[]
        datanumber_cnt=0 
        label_numbers=np.array(self.data_cnt.values())
        train_number_array=np.array(
            [int(i*self.split_ratio['train']) for i in label_numbers.tolist()])
        val_number_array=np.array(
            [int(i*self.split_ratio['val']) for i in label_numbers.tolist()])
        
        for person_file in os.listdir(self.src_path):
            eye_file_name = os.path.join(self.src_path, person_file)
            
            if not os.path.isdir(eye_file_name):
                continue
            for eye_file in os.listdir(eye_file_name):
                # OS/OD
                file_dic = os.path.join(self.src_path, person_file, eye_file)
                if not os.path.isdir(file_dic):
                    continue
                
                file_class_cnt=dict(zip(self.data_cnt.keys(),[0]*(self.class_number)))
                annotations=[]

                for file in os.listdir(file_dic):
                    # if the data can be used
                    if not file.endswith(".jpg"):
                        continue 
                    try:
                        image=Image.open(os.path.join(file_dic,file))
                    except:
                        # print("{} can not open".format(
                        #     os.path.join(file_dic,file)))
                        continue
                    # generate vessel and saved
                    label = self.get_label(file,file_dic,logger=False)
                    file_name=self.get_map_name(os.path.join(person_file,eye_file,file))
                    
                    if label=="-1":
                        # unexpectedd stage
                        continue
                        
                    # Build dict to record the label cnt
                    file_class_cnt[label]+=1

                    shutil.copy(os.path.join(file_dic,file),
                        os.path.join(self.tar_path, 'images',file_name))
                    annotations.append(self.get_json(datanumber_cnt,
                                                     image_name=file_name,
                                                     image_name_original= file,
                                                     image_path=os.path.join(self.tar_path, 'images',file_name),
                                                     image_path_original = os.path.join(file_dic,file),
                                                     label=label))# the label now just act as Placeholder
                    datanumber_cnt+=1
                
                    
                # because of Note 2 
                file_label_number=np.array([i for i in file_class_cnt.values()])
                if np.min(train_number_array-file_label_number)>=0:
                    train_number_array-=file_label_number
                    train_annotations.extend(annotations)
                elif np.min(val_number_array-file_label_number)>=0:
                    val_number_array-=file_label_number
                    val_annotations.extend(annotations)
                else:
                    test_annotations.extend(annotations)
    
        # store annotation in Json file
        with open(os.path.join(self.tar_path, 'annotations', "train.json"), 'w') as f:
            json.dump(train_annotations, f)
        with open(os.path.join(self.tar_path, 'annotations', "val.json"), 'w') as f:
            json.dump(val_annotations, f)
        with open(os.path.join(self.tar_path, 'annotations', "test.json"), 'w') as f:
            json.dump(test_annotations, f)

        # print the data condition
        train_number_array_original=np.array(
            [int(i*self.split_ratio['train']) for i in label_numbers.tolist()])
        val_number_array_original=np.array(
            [int(i*self.split_ratio['val']) for i in label_numbers.tolist()])
        
        train_array=train_number_array_original-train_number_array
        val_array=val_number_array_original-val_number_array
        test_array=np.array([i for i in label_numbers.tolist()])-train_array-val_array
        print("-------Dataset Split Condition-------")
        print("Train:")
        print(dict(zip(self.data_cnt.keys(),train_array)))
        print("Val:")
        print(dict(zip(self.data_cnt.keys(),val_array)))
        print("Test:")
        print(dict(zip(self.data_cnt.keys(),test_array)))

    

if __name__ == '__main__':
    # Init the args
    args = get_config()
    if args.generate_ridge:
        from utils_ import generate_ridge
        generate_ridge(args.json_file_dict,args.path_tar)
    if args.cleansing:
        cleansing_processer=generate_data_processer(src_path=args.path_src,
                                                    tar_path=args.path_tar,
                                                    spilt_train=args.train_split,
                                                    spilt_val=args.val_split )
        cleansing_processer.paser()
    
    # if args.data_augument:
    #     generate_data_augument(data_path=args.path_tar)