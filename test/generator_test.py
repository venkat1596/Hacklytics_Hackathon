"""
This is a test code for generator class

"""

import sys
import os
# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import Unet

def test_unet():
    model = Unet(1, 1, 16)

    image = torch.rand(1,1, 150, 256, 256)

    out = model(image)

    print(out.shape)



if __name__ == "__main__":
    test_unet()