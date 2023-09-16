from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import os
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), 1)


class UNet(nn.Module):

    def __init__(self, configs):
        super(UNet, self).__init__()
        os.environ['TORCH_HOME']=configs["official_model_save"]
        self.C_embed = Conv(configs["in_channels"], 16)
        self.D_embed = DownSampling(16)
        self.C0 = Conv(16, 32)
        self.D0 = DownSampling(32)
        self.C1 = Conv(32, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)
          # Replace your old classifier with a resnet18-like classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Pooling
            nn.Flatten(),  # Flatten the tensor before fully connected layers
            nn.Linear(512, 256),  # Fully connected layer
            nn.ReLU(inplace=True),  # Activation
            nn.Dropout(0.5),  # Dropout for regularization
            # Fully connected layer for num_classes output
            nn.Linear(256, configs['num_classes']),
        )

    def forward(self, x):
        embed = self.C_embed(x)
        R0 = self.C0(self.D_embed(embed))
        R1 = self.C1(self.D0(R0))
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))  # 1024

        O1 = self.C6(self.U1(Y1, R4))  # 512
        O2 = self.C7(self.U2(O1, R3))  # 256
        O3 = self.C8(self.U3(O2, R2))  # 128
        O4 = self.C9(self.U4(O3, R1))  # 64

        heatmap = self.Th(self.pred(O4)).squeeze()  # segment
        distance = self.classifier(O1)
        
        return (heatmap, distance)


class Loss_Unet():
    def __init__(self, locat_r=0.7,Loss_func="MSELoss"):
        self.r = locat_r
        # DiceLoss,FocalLoss
        self.location_loss = getattr(nn,Loss_func)()
        self.class_loss = nn.CrossEntropyLoss()

    def __call__(self, ouputs, targets):
        out_heatmap, out_distance = ouputs
        gt_heatmap, gt_distance = targets
        return self.r*self.class_loss(out_heatmap, gt_heatmap) + \
            (1-self.r)*self.class_loss(out_distance, gt_distance)

def Build_UNet(config):

    model = UNet(config)
    # pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    # model.init_weights(pretrained=pretrained)
    return model, Loss_Unet(locat_r=config["location_r"],Loss_func=config["loss_func"])


if __name__ == "__main__":
    # Initialize the model
    model, citeria = Build_UNet(in_channels=1)

    # Print the model architecture
    print(model)
