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


class AntiAliasDownsample(nn.Module):
    """Anti-aliased downsampling inspired by ResnetGenerator"""

    def __init__(self, channels):
        super().__init__()
        # Blur kernel similar to ResnetGenerator's Downsample
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        # Apply blur kernel then subsample
        x = F.conv2d(self.pad(x), self.kernel, stride=2, groups=c)
        return x


class ImprovedDownBlock(nn.Module):
    """Downsampling block with anti-aliasing and spectral normalization"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)),
            nn.InstanceNorm2d(in_ch),
            nn.ReLU(True)
        )
        self.conv2 = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(True)
        self.downsample = AntiAliasDownsample(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.act(x)
        return self.downsample(x)


class ResidualDilatedSEBlock(nn.Module):
    """Improved dilated block with stronger residual connection and normalization"""

    def __init__(self, channels, dilation=2):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = spectral_norm(nn.Conv2d(channels, channels, 3,
                                             padding=dilation, dilation=dilation))
        self.norm2 = nn.InstanceNorm2d(channels)
        self.se = SqueezeExcitation(channels, reduction=8)  # Stronger reduction
        self.act = nn.ReLU(True)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.se(out)
        out = self.act(out)
        return residual + out  # Clean residual connection


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


class AntiAliasUpsample(nn.Module):
    """Anti-aliased upsampling for better image quality"""

    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Smooth after upsampling
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(channels, channels, kernel_size=3, padding=0, groups=channels)),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(self.upsample(x))


class ImprovedUpBlock(nn.Module):
    """Upsampling block with anti-aliasing and improved connections"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = AntiAliasUpsample(in_ch)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_ch + in_ch, out_ch, 3)),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EnhancedFusionBlock(nn.Module):
    """Improved fusion with direct residual paths and better normalization"""

    def __init__(self, low_ch, high_ch):
        super().__init__()
        self.process_low = nn.Sequential(
            spectral_norm(nn.Conv2d(low_ch, high_ch, 1)),
            nn.InstanceNorm2d(high_ch),
            nn.ReLU(True)
        )
        # Two-stage attention for more precise feature selection
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            spectral_norm(nn.Conv2d(high_ch, high_ch // 8, 1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(high_ch // 8, high_ch, 1)),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, low, high):
        # Align spatial dimensions
        aligned_low = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=False)
        processed_low = self.process_low(aligned_low)

        # Channel attention
        channel_weights = self.channel_attn(high)
        high_weighted = high * channel_weights

        # Spatial attention
        avg_pool = torch.mean(high_weighted, dim=1, keepdim=True)
        max_pool, _ = torch.max(high_weighted, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attn(spatial_input)

        # Combine features with dual attention
        fused = high_weighted * spatial_weights + processed_low * (1 - spatial_weights)
        return fused


class MultiScaleFusionGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, n_blocks=6):
        super().__init__()
        # Initial convolution with larger kernel (like ResnetGenerator)
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),  # Better boundary handling
            spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)),
            nn.InstanceNorm2d(ngf),  # Instance norm works well for style transfer
            nn.ReLU(True)  # Starting with ReLU
        )

        # Encoder
        self.down1 = ImprovedDownBlock(ngf, ngf*2)
        self.down2 = ImprovedDownBlock(ngf*2, ngf*4)
        self.down3 = ImprovedDownBlock(ngf*4, ngf*4)

        # Dilated Residual Blocks with SE
        self.res_blocks = nn.Sequential(*[
            ResidualDilatedSEBlock(ngf * 4, dilation=2 ** (i % 3))
            for i in range(n_blocks)
        ])

        # Decoder
        self.up1 = ImprovedUpBlock(ngf*4, ngf*2)
        self.up2 = ImprovedUpBlock(ngf*2, ngf)
        self.up0 = ImprovedUpBlock(ngf, ngf)

        # Multi-Scale Fusion Modules
        self.fuse1 = EnhancedFusionBlock(ngf, ngf * 2)
        self.fuse2 = EnhancedFusionBlock(ngf * 2, ngf * 4)
        self.fuse3 = EnhancedFusionBlock(ngf * 4, ngf * 4)

        # Output
        # Refinement block before final output
        self.refine = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        # Output layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )
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

        x3 = self.down3(x2)
        x3 = self.res_blocks(x3)
        if 3 in layers:
            features.append(x3)

        if encode_only:
            return features

        # Multi-Scale Fusion
        f1 = self.fuse1(x0, x1) + x1
        f2 = self.fuse2(x1, x2) + x2
        f3 = self.fuse3(x2, x3) + x3

        # Decoder Path
        x = self.up1(f3, f2)
        x = self.up2(x, f1)
        x = self.up0(x, x0)

        # Final refinement
        x = self.refine(x) + x  # Final residual connection

        return self.tanh(self.final(x))








