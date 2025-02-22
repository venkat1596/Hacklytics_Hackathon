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

    def forward(self, x):
        x = self.act(self.first_norm(self.conv_first(x)))
        return x

class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.pad = nn.ReflectionPad3d(1)
        self.conv = nn.Conv3d(features, features, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(self.pad(x))
        return x

class PatchDisc(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()

        class PatchDisc(nn.Module):
            def __init__(self, in_channels, features):
                super().__init__()
                # Reduce to 3 blocks instead of 4
                self.conv1 = ConvModule(in_channels, features)
                self.down1 = Downsample(features)

                self.conv2 = ConvModule(features, features * 2)
                self.down2 = Downsample(features * 2)

                self.conv3 = ConvModule(features * 2, features * 2)  # Limit feature multiplication
                self.down3 = Downsample(features * 2)

                self.final_conv = nn.Conv3d(features * 2, features, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)

        x = self.conv2(x)
        x = self.down2(x)

        x = self.conv3(x)
        x = self.down3(x)

        x = self.final_conv(x)
        return x

    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad



class SpectralNormConv3d(nn.Module):
    """Convolution layer with spectral normalization for stability"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_act=True):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
            )
        if use_act:
            self.act = Smish()
        self.use_act = use_act

    def forward(self, x):
        x = self.conv(x)
        if self.use_act:
            x = self.act(x)
        return x

class Spectral_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            SpectralNormConv3d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNormConv3d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNormConv3d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNormConv3d(128, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.conv_blocks(x)

