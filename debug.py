import torch,os
from VesselSegModule.models import FR_UNet
from VesselSegModule  import VesselSegProcesser
# Load checkpoin
processer=VesselSegProcesser('./model_save')
data_list=[f"{str(i+1)}.jpg" for i in range(40)]
for image_name in data_list:
    processer(os.path.join('../autodl-tmp/dataset_ROP/images',image_name),
              os.path.join('./experiments',image_name))