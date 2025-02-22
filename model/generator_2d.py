import torch
import torch.nn as nn
import torch.nn.functional as F

from .Xsmish import Smish


class Simple2DSAFM(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // ratio

        # 2D versions of SAFM components
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.dwconv = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 1,
                                groups=self.chunk_dim, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.act = Smish()

    def forward(self, x):
        h, w = x.size()[-2:]
        x0, x1 = self.proj(x).split([self.chunk_dim, self.dim - self.chunk_dim], dim=1)

        # 2D adaptive pooling and upsampling
        x2 = F.adaptive_max_pool2d(x0, (h // 4, w // 4))
        x2 = self.dwconv(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')
        x2 = self.act(x2) * x0

        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))
        return x


class CCM2D(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        hidden_dim = int(dim * ffn_scale)
        self.conv1 = nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        self.act = Smish()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class InvertibleAttBlock2D(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm = nn.InstanceNorm2d(dim)
        self.safm = Simple2DSAFM(dim, ratio=4)
        self.ccm = CCM2D(dim, ffn_scale)

        # Invertibility components
        self.conv_residual = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
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


class EfficientInvertibleGenerator2D(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_blocks=8, ffn_scale=2.0):
        super().__init__()

        # Initial feature extraction
        self.to_feat = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim),
            Smish()
        )

        # Main processing blocks
        self.blocks = nn.ModuleList([
            InvertibleAttBlock2D(dim, ffn_scale)
            for _ in range(n_blocks)
        ])

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            Smish(),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid()
        )

        # Output projection
        self.to_out = nn.Sequential(
            nn.Conv2d(dim, in_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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