import sys
import os
# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import PatchSampleF, MultiScaleFusionGenerator


def test_patch_sample_f():
    model_f = PatchSampleF(use_mlp=True, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[])
    model_g = MultiScaleFusionGenerator(input_nc=1, ngf=32, output_nc=1)
    image = torch.rand(1, 1, 256, 256)

    feat_g = model_g(image, layers=[0, 1, 2, 3], encode_only=True)

    feat_g, patch_ids = model_f(feat_g, num_patches=64)

    print(f"Patch ids: {patch_ids}")
    for fg in feat_g:
        print(f"Feature shape: {fg.shape}")

if __name__ == "__main__":
    test_patch_sample_f()