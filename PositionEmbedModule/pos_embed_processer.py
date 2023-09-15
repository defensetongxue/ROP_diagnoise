# this file will create an interface for the rop_dig
import torch
import torchvision.transforms as transforms
from PIL import Image
import inspect
import os,json
from .models import ViT
def get_instance(module, class_name, *args, **kwargs):
    try:
        cls = getattr(module, class_name)
        instance = cls(*args, **kwargs)
        return instance
    except AttributeError:
        available_classes = [name for name, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__]
        raise ValueError(f"{class_name} not found in the given module. Available classes: {', '.join(available_classes)}")


class PosEmbedProcesser():
    def __init__(self, split_name="mini",config_path="./config_file/pos_embed.json",
                 model_dict="./ROP_diagnoise/model_save"):
        with open(config_path,'r') as f:
            cfgs=json.load(f)
        self.model = ViT(cfgs['model'])
        checkpoint = torch.load(
            os.path.join(model_dict,f'{split_name}_pos_embed.pth'))
        self.model.load_state_dict(checkpoint)
        self.model.cuda()

        # generate mask
        self.transforms = transforms.Compose([
            transforms.Resize(cfgs['image_resize']),
            transforms.ToTensor()
        ])

    def __call__(self, vessel_path,save_path=None):
        # open the image and preprocess
        vessel=Image.open(vessel_path).convert('RGB')
        img = self.transforms(vessel)

        # generate predic vascular with pretrained model
        img = img.unsqueeze(0)  # as batch size 1
        pre = self.model(img.cuda())
        predict = torch.sigmoid(pre).cpu().detach().squeeze()

        if save_path:
            torch.save(predict,save_path)
        return predict.numpy()
