# this file will create an interface for the rop_dig
from . import models
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import inspect
def get_instance(module, class_name, *args, **kwargs):
    try:
        cls = getattr(module, class_name)
        instance = cls(*args, **kwargs)
        return instance
    except AttributeError:
        available_classes = [name for name, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__]
        raise ValueError(f"{class_name} not found in the given module. Available classes: {', '.join(available_classes)}")


class PosEmbedProcesser():
    def __init__(self, model_name,
                 vessel_resize,image_orignal_size,patch_size):
        self.model = get_instance(models, model_name,
                    patch_size=patch_size,
                    image_size=vessel_resize,
                    embed_dim=64,
                     depth=3,
                     heads=4,
                     mlp_dim=32,
                    #  dropout=0.
                     )
        checkpoint = torch.load(
            './PositionEmbedModule/checkpoint/pos_embed.pth')
        self.model.load_state_dict(checkpoint)
        self.model.cuda()

        self.patch_size=patch_size
        # generate mask
        self.image_original_size=image_orignal_size
        self.transforms = transforms.Compose([
            transforms.Resize((vessel_resize,vessel_resize)),
            transforms.ToTensor()
        ])

    def __call__(self, vessel,save_path=None):
        # open the image and preprocess
        img = self.transforms(vessel)

        # generate predic vascular with pretrained model
        img = img.unsqueeze(0)  # as batch size 1
        pre = self.model(img.cuda())
        predict = torch.sigmoid(pre).cpu().detach().squeeze()

        if save_path:
            torch.save(predict,save_path)
        return predict.numpy()
