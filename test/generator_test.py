"""
This is a test code for generator class

"""

import sys
import os
# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import Unet, EfficientInvertibleGenerator3D, MultiScaleFusionGenerator

def test_unet():
    model = Unet(1, 1, 16)

    image = torch.rand(1,1, 150, 256, 256)

    out = model(image)

    print(out.shape)


def test_efficient_invertible_generator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientInvertibleGenerator3D(1, 16, 4).to(device)

    image = torch.rand(1,1, 150, 128, 128).to(device)

    out = model(image)

    print(out.shape)

    del out, image, model
    torch.cuda.empty_cache()

def test_multiscale_fusion_generator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiScaleFusionGenerator(input_nc=1, ngf=32, output_nc=1).to(device)

    image = torch.rand(1,1, 256, 256).to(device)

    layers = [0, 1, 2, 3]
    out = model(image)

    encode_out = model(image, layers=layers, encode_only=True)

    print(out.shape)

    for i, encode in enumerate(encode_out):
        print(f"Layer {i} shape: {encode.shape}")

    del out, image, model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # test_unet()
    # test_efficient_invertible_generator()
    test_multiscale_fusion_generator()