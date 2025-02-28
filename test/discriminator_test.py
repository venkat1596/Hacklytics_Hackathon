import sys
import os
# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import NLayerDiscriminator

if __name__ == '__main__':
    image = torch.randn(1, 1, 256, 256)
    model = NLayerDiscriminator(1, 32, 3)
    out = model(image)

    print(out.shape)

