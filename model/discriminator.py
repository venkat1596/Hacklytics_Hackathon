from xml.sax.handler import feature_string_interning

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Xsmish import Smish


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    padding=kernel_size//2)
        self.first_norm = nn.InstanceNorm2d(out_channels)
        self.act = Smish()

    def forward(self, x):
        x = self.act(self.first_norm(self.conv_first(x)))
        return x

class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv = nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
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


import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class NLayerDiscriminator(nn.Module):
    """Spectral Normalized PatchGAN Discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, no_antialias=False):
        super().__init__()
        kw = 4
        padw = 1

        # Layer sequence construction
        layers = []

        # Initial block
        if no_antialias:
            conv = spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
            layers += [conv, nn.LeakyReLU(0.2, True)]
        else:
            conv = spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw))
            layers += [conv, nn.LeakyReLU(0.2, True), Downsample(ndf)]

        # Intermediate blocks
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                block = [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                            kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                block = [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                            kernel_size=kw, stride=1, padding=padw)),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)
                ]
            layers += block

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        # Output layer
        layers += [
            spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x