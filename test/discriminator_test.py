import sys
import os
# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import Discriminator

if __name__ == '__main__':
    image = torch.randn(1, 1, 150, 256, 256)
    model = Discriminator(1, 16)
    out = model(image)

    print(out.shape)

