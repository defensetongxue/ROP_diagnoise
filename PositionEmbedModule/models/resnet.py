from einops import rearrange, repeat
import torch.nn.functional as F

import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        assert configs["image_resize"][0]==configs["image_resize"][1]
        assert configs["image_resize"][0] % configs["patch_size"] == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_length = int(configs["image_resize"][0]/configs["patch_size"])
        self.patch_size = configs["patch_size"]
        self.patch_embedding = PatchEmbedding(in_channels=configs["in_channels"],
                                              patch_size=configs["patch_size"],
                                              embed_dim=configs["embed_dim"])
        self.dropout = nn.Dropout(configs["emb_dropout"])

        self.backbone=blackbone(embed_dim=configs["embed_dim"],
                           depth=configs["depth"])
        
        self.classifier=nn.Sequential(
            nn.BatchNorm2d(configs["embed_dim"]),
            nn.Conv2d(configs["embed_dim"],32,1),
            nn.Conv2d(32,1,1)
        )
        
    def forward(self, img):
        x = self.patch_embedding(img)
        x = self.dropout(x)
        x= self.backbone(x)
        x=self.classifier(x)
        return x.squeeze(1)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x) 
        # shape = [batch_size, embed_dim, num_patches ** 0.5, num_patches ** 0.5]
        return x

class blackbone(nn.Module):
    def __init__(self, embed_dim=64, depth=3):
        super().__init__()
        layers=[]
        for _ in range(depth):
            layers.append(conv(
                embed_dim,embed_dim,0.2
            ))
        self.convlayer=nn.Sequential(*layers)
    def forward(self,x):
        x=self.convlayer(x)
        return x
class conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res = x
        x = self.conv(x)
        out = x + res
        out = self.relu(out)
        return x
if __name__ =='__main__':
    configs={
        "image_resize":[256,256],
        "patch_size":32,
        "embed_dim":64,
        "in_channels":3,
        "emb_dropout":0.2,
        "depth":3
    }
    input_tensor = torch.rand(3, 3, 256, 256)
    model=ResNet(configs)
    output_tensor=model(input_tensor)
    print(output_tensor.shape)