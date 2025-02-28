import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .Fsmish import smish as Fsmish
from .Xsmish import Smish


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = max(4, in_channels // reduction)  # Ensure minimum 4 channels

        self.se = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, reduced_channels, 1)),
            Smish(),
            spectral_norm(nn.Conv2d(reduced_channels, in_channels, 1)),
            nn.Sigmoid()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x)
        y = self.se(y)
        return x * y.expand_as(x)


class DilatedSEBlock(nn.Module):
    """Dilated convolution with SE and residual"""

    def __init__(self, channels, dilation=2):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3,
                                    padding=dilation, dilation=dilation)),
            SqueezeExcitation(channels),
            Smish()
        )

    def forward(self, x):
        return x + self.conv(x)


class DepthSepHybridBlock(nn.Module):
    """Combines depthwise conv, SE, and dilation"""

    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3,
                                    padding=dilation, dilation=dilation, groups=in_ch)),
            spectral_norm(nn.Conv2d(in_ch, out_ch, 1)),
            SqueezeExcitation(out_ch),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class DepthSepUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch * 2, in_ch, 3,
                          padding=1, groups=in_ch)),
            spectral_norm(nn.Conv2d(in_ch, out_ch, 1)),
            Smish()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.upsample(x)
        x = self.conv(x)
        return x


class MultiScaleFusionBlock(nn.Module):
    """Fuses features from different scales"""

    def __init__(self, low_ch, high_ch):
        super().__init__()
        self.process_low = nn.Sequential(
            spectral_norm(nn.Conv2d(low_ch, high_ch, 1)),
            Smish()
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            spectral_norm(nn.Conv2d(high_ch, high_ch // 8, 1)),
            Smish(),
            spectral_norm(nn.Conv2d(high_ch // 8, high_ch, 1)),
            nn.Sigmoid()
        )

    def forward(self, low, high):
        aligned_low = F.interpolate(low, size=high.shape[2:], mode='bilinear')
        processed_low = self.process_low(aligned_low)
        attn = self.attention(high)
        fused = high * attn + processed_low * (1 - attn)
        return Fsmish(fused)


class MultiScaleFusionGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, n_blocks=6):
        super().__init__()
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ngf, 7, padding=3)),
            Smish()
        )

        # Encoder
        self.down1 = DepthSepHybridBlock(ngf, ngf*2)
        self.down2 = DepthSepHybridBlock(ngf*2, ngf*4)

        # Dilated Residual Blocks with SE
        self.res_blocks = nn.Sequential(*[
            DilatedSEBlock(ngf * 4, dilation=2 ** (i % 3))
            for i in range(n_blocks)
        ])

        # Decoder
        self.up1 = DepthSepUpBlock(ngf*4, ngf*2)
        self.up2 = DepthSepUpBlock(ngf*2, ngf)

        # Multi-Scale Fusion Modules
        self.fuse1 = MultiScaleFusionBlock(ngf, ngf * 2)
        self.fuse2 = MultiScaleFusionBlock(ngf * 2, ngf * 4)
        self.fuse3 = MultiScaleFusionBlock(ngf * 4, ngf * 4)

        # Output
        self.final = spectral_norm(nn.Conv2d(ngf, output_nc, 7, padding=3))
        self.tanh = nn.Tanh()

    def forward(self, x, layers=[], encode_only=False):
        features = []
        x0 = self.initial(x)
        if 0 in layers:
            features.append(x0)

        x1 = self.down1(x0)
        if 1 in layers:
            features.append(x1)

        x2 = self.down2(x1)
        if 2 in layers:
            features.append(x2)

        x3 = self.res_blocks(x2)
        if 3 in layers:
            features.append(x3)

        if encode_only:
            return features

        # Multi-Scale Fusion
        f1 = self.fuse1(x0, x1)
        f2 = self.fuse2(x1, x2)
        f3 = self.fuse3(x2, x3)

        # Decoder Path
        x = self.up1(f3, f2)
        x = self.up2(x, f1)

        return self.tanh(self.final(x))








