import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3, stride=2)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class ContrastiveDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, features, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            Downsample(features)
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(features, features, kernel_size=3, stride=2)
        ]
        features_prev = features
        for i in range(3):
            features *= 2
            layers += [
                nn.Conv2d(features_prev, features, kernel_size=4, stride=1, padding=1),
                nn.InstanceNorm2d(features),
                nn.LeakyReLU(0.2, True)
            ]
            features_prev = features
            if i<2:
                layers += [
                    Downsample(features)
                    # nn.ReflectionPad2d(1),
                    # nn.Conv2d(features, features, kernel_size=3, stride=2)
                ]
        features = 1
        layers += [
            nn.Conv2d(features_prev, features, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return(self.model(input))

    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad