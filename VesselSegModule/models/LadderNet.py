#implement by chatgpt
import torch
import torch.nn as nn
import torch.nn.functional as F

class LadderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LadderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out_down = self.pool(out)
        out_up = F.interpolate(out_down, size=out.shape[-2:], mode='bilinear', align_corners=True)
        out_up = self.bn2(self.conv2(out_up))
        out = out + out_up
        return out, out_down

class LadderNet(nn.Module):
    def __init__(self, num_classes):
        super(LadderNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.block1 = LadderBlock(64, 128)
        self.block2 = LadderBlock(128, 256)
        self.block3 = LadderBlock(256, 512)
        self.block4 = LadderBlock(512, 1024)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_last = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, out_down1 = self.block1(out)
        out, out_down2 = self.block2(out)
        out, out_down3 = self.block3(out)
        out, out_down4 = self.block4(out)
        out = F.relu(self.upconv1(out))
        out = torch.cat([out, out_down4], dim=1)
        out = F.relu(self.upconv2(out))
        out = torch.cat([out, out_down3], dim=1)
        out = F.relu(self.upconv3(out))
        out = torch.cat([out, out_down2], dim=1)
        out = F.relu(self.upconv4(out))
        out = torch.cat([out, out_down1], dim=1)
        out = self.conv_last(out)
        return out
