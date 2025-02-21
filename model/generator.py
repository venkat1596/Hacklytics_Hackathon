import torch
import torch.nn as nn
import torch.nn.functional as F

from .Xsmish import Smish
from .Fsmish import smish as Fsmish


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv3d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv3d(mid_features, out_features, 3, padding=1)
        self.norm = nn.InstanceNorm3d(mid_features)
        self.smish= Smish()#nn.ReLU(inplace=True)

    def forward(self, x):
        x_copy = x.clone()
        x = self.conv1(x)
        x = self.norm(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        x += x_copy
        return x


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                    padding=kernel_size//2)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = Smish()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x

class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.pad = nn.ReflectionPad3d(1)
        self.conv = nn.Conv3d(features, features, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(self.pad(x))
        return x

class DilatedConvBlock(nn.Module):
    def __init__(self, features, no_of_layers, dilation=4):
        super(DilatedConvBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(no_of_layers):
            self.layers.append(nn.Conv3d(features, features, 3, padding=dilation, dilation=dilation))
            self.layers.append(nn.InstanceNorm3d(features))
            self.layers.append(Smish())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = in_features
            layers.append(nn.Conv3d(in_features, out_features, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose3d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super().__init__()

        # Encoder
        # block 1
        self.conv1 = SingleConv(in_channels, features)

        # block 2
        self.conv2 = DoubleConvBlock(features, features * 2)
        self.down2 = Downsample(features * 2)

        # block 3
        self.conv3 = DilatedConvBlock(features*2, 3, dilation=2)
        self.down3 = Downsample(features*2)

        # block 4
        self.conv4 = DilatedConvBlock(features*2, 3, dilation=2)
        # Decoder

        # up blocks
        self.upblock3 = UpConvBlock(features*2, 1)
        self.upblock2 = UpConvBlock(features*2, 1)
        self.correctionblock2 = SingleConv(features*2, features)
        self.upblock1 = UpConvBlock(features, 1)



        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2_skip = x1.clone()
        x2 = self.down2(x2)

        x3 = self.conv3(x2)
        x3_skip = x3.clone()
        x3 = self.down3(x3)

        x4 = self.conv4(x3)

        x3_up = self.upblock3(x4)
        x3_up = torch.cat([x3_skip, x3_up], dim=1)







