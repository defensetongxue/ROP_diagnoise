# this file will create an interface for the rop_dig
from . import models
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
def decompose_image_into_tensors(image):
    # Assumes the input is a PyTorch tensor
    height, width = image.shape[1], image.shape[2]
    # Split the image tensor into two along the height
    first_half, second_half = torch.split(image, height//2, dim=1)
    # Then split each half into two along the width
    first_half_tensors = torch.split(first_half, width//2, dim=2)
    second_half_tensors = torch.split(second_half, width//2, dim=2)
    # Return a list of all four image parts
    return list(first_half_tensors) + list(second_half_tensors)


def compose_tensors_into_image(tensors_list):
    # Assumes the input is a list of four tensors
    top = torch.cat(tensors_list[:2], dim=1)  # Concatenate along width
    bottom = torch.cat(tensors_list[2:], dim=1)  # Concatenate along width
    return torch.cat([top, bottom], dim=0)  # Concatenate along height


class VesselSegProcesser():
    def __init__(self, model_name,
                 resize=(512, 512)):
        self.model = getattr(models, model_name)()
        checkpoint = torch.load(
            './VesselSegModule/checkpoint/best.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()

        self.resize = resize
        # generate mask
        mask = Image.open('./VesselSegModule/mask.png')
        mask = transforms.ToTensor()(mask)[0]
        self.mask = mask

        self.transforms = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.3968], [0.1980])
            # the mean and std is cal by 12 rop1 samples
            # TODO using more precise score
        ])

    def __call__(self, img,save_path=None):
        # open the image and preprocess
        with torch.no_grad():
            decomposed_images = decompose_image_into_tensors(self.transforms(img))
            composed_mask = []
            for decomposed_image in decomposed_images:
                decomposed_image = decomposed_image.unsqueeze(0).cuda() 
                mask_part = self.model(decomposed_image).detach().cpu()
                mask_part = torch.sigmoid(mask_part.squeeze())
                composed_mask.append(mask_part)

            vessel = compose_tensors_into_image(composed_mask)
            # mask
            predict = torch.where(self.mask < 0.1, self.mask, vessel)
            if save_path:
                cv2.imwrite(save_path,
                    np.uint8(predict.numpy()*255))
            return predict.numpy()
