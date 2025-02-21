from xml.sax.handler import feature_string_interning

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Xsmish import Smish


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_first = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                    padding=kernel_size//2)
        self.first_norm = nn.InstanceNorm3d(out_channels)
        self.act = Smish()

        self.conv_second = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                                    padding=kernel_size//2)
        self.second_norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x = self.act(self.first_norm(self.conv_first(x)))
        x = self.act(self.second_norm(self.conv_second(x)))
        return x

class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.pad = nn.ReflectionPad3d(1)
        self.conv = nn.Conv3d(features, features, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(self.pad(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()

        # block 1
        self.conv1 = ConvModule(in_channels, features)
        self.down1 = Downsample(features)

        # block 2
        self.conv2 = ConvModule(features, features*2)
        self.down2 = Downsample(features*2)

        # block 3
        self.conv3 = ConvModule(features*2, features*4)
        self.down3 = Downsample(features*4)

        # block 4
        self.conv4 = ConvModule(features*4, features*4)
        self.down4 = Downsample(features*4)


        self.final_conv = nn.Conv2d(features*4, features, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)

        x = self.conv2(x)
        x = self.down2(x)

        x = self.conv3(x)
        x = self.down3(x)

        x = self.conv4(x)
        x = self.down4(x)

        x = self.final_conv(x)
        return x

    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad


