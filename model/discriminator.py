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


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=16):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model = nn.Sequential(
            ConvModule(features, features * 2),
            Downsample(features * 2),
            ConvModule(features * 2, features * 4),
            Downsample(features * 4),
            ConvModule(features * 4, features * 4),
            nn.Conv2d(features * 4, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.initial(x)
        return self.model(x)



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

