import torch
import torch.nn as nn
import torch.nn.functional as F

from .Xsmish import Smish
from .Fsmish import smish as Fsmish
from .pixelshuffle_3d import PixelShuffle3d

def weight_init(m):
    if isinstance(m, (nn.Conv3d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose3d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

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
        x = self.conv1(x)
        x = self.norm(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
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
        self.conv = nn.Conv3d(features, features, kernel_size=3, stride=(1, 2, 2))

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
        x_copy = x.clone()
        x = self.layers(x)
        x += x_copy
        return x

class Upsampler(nn.Module):
    def __init__(self, features, upscale, stride, output_padding):
        super().__init__()
        layers = []
        for _ in range(upscale):
            layers.append(nn.Conv3d(features, features, 3, padding=1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose3d(features, features, kernel_size=3, padding=1, output_padding=output_padding,
                                             stride=stride))
        self.conv_t = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_t(x)
        return x

class DoubleFusion(nn.Module):
    def __init__(self, in_ch):
        super(DoubleFusion, self).__init__()
        self.DWconv1 = nn.Conv3d(in_ch, in_ch*8, kernel_size=3,
                               stride=1, padding=1, groups=in_ch)
        self.PSconv1 = PixelShuffle3d(1)
        self.DWconv2 = nn.Conv3d(in_ch*8, in_ch*8, kernel_size=3,
                               stride=1, padding=1, groups=in_ch*8)
        self.final_conv = nn.Conv3d(in_ch*8, in_ch, kernel_size=3,
                                  stride=1, padding=1, groups=in_ch)
        self.AF = Smish()

    def forward(self, x):
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return Fsmish(self.final_conv(attn2 + attn))

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super().__init__()

        # Encoder
        # block 1
        self.conv1 = DoubleConvBlock(in_channels, features, stride=2)

        # block 2
        self.conv2 = DilatedConvBlock(features, 3, dilation=2)
        self.down2 = Downsample(features)

        # block 3
        self.conv3 = DilatedConvBlock(features, 3, dilation=2)
        # Decoder

        # up blocks
        self.upblock3 = Upsampler(features, 1, stride=(1, 2, 2), output_padding=(0, 1, 1))
        self.upblock2 = Upsampler(features, 1, stride=(2, 2, 2), output_padding=(1, 1, 1))

        # decoder fuse block
        self.fuse_block3 = DoubleFusion(features * 2)
        self.fuse_conv3 = SingleConv(features * 2, features)

        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2_skip = x1.clone()
        x2 = self.down2(x2)

        x3 = self.conv3(x2)
        x3_skip = x3.clone()
        x3 += x3_skip

        x2_up = self.upblock3(x3)
        x2_up = torch.cat([x2_skip, x2_up], dim=1)
        x2_fuse = self.fuse_block3(x2_up)
        x2_fuse = self.fuse_conv3(x2_fuse)

        x1_up = self.upblock2(x2_fuse)

        x1 = self.final_conv(x1_up)
        x1 = self.final_act(x1)

        return x1

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


class DoubleSpectralNormConv2d(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv1 = SpectralNormConv3d(in_channels, mid_channels, kernel_size, stride, padding, use_act=True)
        self.conv2 = SpectralNormConv3d(mid_channels, out_channels, kernel_size, stride, padding, use_act=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Simple3DSAFM(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // ratio

        # 3D adaptations of SAFM components
        self.proj = nn.Conv3d(dim, dim, 3, 1, 1, bias=False)
        self.dwconv = nn.Conv3d(self.chunk_dim, self.chunk_dim, 3, 1, 1,
                                groups=self.chunk_dim, bias=False)
        self.out = nn.Conv3d(dim, dim, 1, 1, 0, bias=False)
        self.act = Smish()

    def forward(self, x):
        d, h, w = x.size()[-3:]
        x0, x1 = self.proj(x).split([self.chunk_dim, self.dim - self.chunk_dim], dim=1)

        # 3D adaptive pooling and upsampling
        x2 = F.adaptive_max_pool3d(x0, (d // 2, h // 4, w // 4))
        x2 = self.dwconv(x2)
        x2 = F.interpolate(x2, size=(d, h, w), mode='trilinear')
        x2 = self.act(x2) * x0

        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))
        return x


class CCM3D(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        hidden_dim = int(dim * ffn_scale)
        self.conv1 = nn.Conv3d(dim, hidden_dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv3d(hidden_dim, dim, 1, 1, 0, bias=False)
        self.act = Smish()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class InvertibleAttBlock3D(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm = nn.InstanceNorm3d(dim)
        self.safm = Simple3DSAFM(dim, ratio=4)
        self.ccm = CCM3D(dim, ffn_scale)

        # Invertibility components
        self.conv_residual = nn.Conv3d(dim, dim, 1, 1, 0, bias=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.act = Smish()

    def forward(self, x):
        identity = x
        out = self.norm(x)
        out = self.safm(out)
        out = self.ccm(out)
        return identity + self.scale * self.act(out)

    def inverse(self, y):
        return (y - self.scale * self.act(self.ccm(self.safm(self.norm(y)))))


class EfficientInvertibleGenerator3D(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_blocks=8, ffn_scale=2.0):
        super().__init__()

        # Initial feature extraction
        self.to_feat = nn.Sequential(
            nn.Conv3d(in_channels, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(dim),
            Smish()
        )

        # Main processing blocks
        self.blocks = nn.ModuleList([
            InvertibleAttBlock3D(dim, ffn_scale)
            for _ in range(n_blocks)
        ])

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv3d(dim, dim // 2, 1),
            Smish(),
            nn.Conv3d(dim // 2, dim, 1),
            nn.Sigmoid()
        )

        # Output projection
        self.to_out = nn.Sequential(
            nn.Conv3d(dim, in_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Feature extraction
        feat = self.to_feat(x)

        # Process through invertible blocks
        for block in self.blocks:
            feat = block(feat)

        # Apply attention
        att = self.attention(feat)
        feat = feat * att

        # Output projection
        return self.to_out(feat)

    def inverse(self, y):
        # Remove output projection
        feat = self.to_feat(y)

        # Remove attention effect approximately
        att = self.attention(feat)
        feat = feat / (att + 1e-6)  # Add small epsilon to prevent division by zero

        # Inverse through blocks in reverse order
        for block in reversed(self.blocks):
            feat = block.inverse(feat)

        return self.to_out(feat)








