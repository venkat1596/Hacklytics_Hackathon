import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=64, n_layers=3, norm_type='instance'):
        """
        N-layer PatchGAN discriminator
        Args:
            in_channels (int): Number of input channels
            features (int): Number of filters in the first conv layer
            n_layers (int): Number of conv layers (minimum 3)
            norm_type (str): Type of normalization ('instance' or 'batch')
        """
        super().__init__()

        # Select normalization layer
        norm_layer = nn.InstanceNorm2d if norm_type == 'instance' else nn.BatchNorm2d

        # Initial layer without normalization
        sequence = [
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        # Calculate current number of features
        nf_mult = 1
        nf_mult_prev = 1

        # Middle layers with increasing number of features
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x the initial features

            sequence += [
                nn.Conv2d(
                    features * nf_mult_prev,
                    features * nf_mult,
                    kernel_size=4,
                    stride=2 if n < n_layers - 1 else 1,  # Last layer uses stride 1
                    padding=1,
                    bias=False
                ),
                norm_layer(features * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final layer
        sequence += [
            nn.Conv2d(
                features * nf_mult,
                1,  # Output one channel prediction map
                kernel_size=4,
                stride=1,
                padding=1
            )
        ]

        self.model = nn.Sequential(*sequence)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.model(x)

    def get_receptive_field(self):
        """Calculate and return the receptive field size of the discriminator"""
        n_layers = len([m for m in self.model if isinstance(m, nn.Conv2d)])
        start_rf = 4  # Initial kernel size
        for i in range(n_layers - 1):
            start_rf = start_rf * 2 + 2  # Each layer doubles RF and adds overlap
        return start_rf


class ProgressivePatchDiscriminator(nn.Module):
    """
    A multi-scale PatchGAN discriminator that operates at multiple scales
    """

    def __init__(self, in_channels=1, features=64, n_scales=3):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PatchDiscriminator(
                in_channels=in_channels,
                features=features,
                n_layers=3 + i  # Progressively deeper discriminators
            ) for i in range(n_scales)
        ])

        # Downsample layer for creating multiple scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)  # Create next scale
        return outputs


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class SimpleDiscriminator(nn.Module):
    # initializers
    def __init__(self, in_channels=1, d=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 4, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.conv4(x)
        return x