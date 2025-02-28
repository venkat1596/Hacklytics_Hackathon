import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


# class PatchSampleF(nn.Module):
#     def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
#         # potential issues: currently, we use the same patch_ids for multiple images in the batch
#         super(PatchSampleF, self).__init__()
#         self.l2norm = Normalize(2)
#         self.use_mlp = use_mlp
#         self.nc = nc  # hard-coded
#         self.mlp_init = False
#         self.init_type = init_type
#         self.init_gain = init_gain
#         self.gpu_ids = gpu_ids
#
#     def create_mlp(self, feats):
#         for mlp_id, feat in enumerate(feats):
#             input_nc = feat.shape[1]
#             mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
#             if len(self.gpu_ids) > 0:
#                 mlp.cuda()
#             setattr(self, 'mlp_%d' % mlp_id, mlp)
#         init_net(self, self.init_type, self.init_gain, self.gpu_ids)
#         self.mlp_init = True
#
#     def forward(self, feats, num_patches=64, patch_ids=None):
#         return_ids = []
#         return_feats = []
#         if self.use_mlp and not self.mlp_init:
#             self.create_mlp(feats)
#         for feat_id, feat in enumerate(feats):
#             B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
#             feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
#             if num_patches > 0:
#                 if patch_ids is not None:
#                     patch_id = patch_ids[feat_id]
#                 else:
#                     # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
#                     #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
#                     patch_id = np.random.permutation(feat_reshape.shape[1])
#                     patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
#                 patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
#                 x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
#             else:
#                 x_sample = feat_reshape
#                 patch_id = []
#             if self.use_mlp:
#                 mlp = getattr(self, 'mlp_%d' % feat_id)
#                 x_sample = mlp(x_sample)
#             return_ids.append(patch_id)
#             x_sample = self.l2norm(x_sample)
#
#             if num_patches == 0:
#                 x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
#             return_feats.append(x_sample)
#         return return_feats, return_ids

class PatchSampleF(nn.Module):
    """PatchSample module that samples patches from features and applies MLP projection.

    This version initializes MLPs right away in the constructor, making it compatible
    with optimizers that need parameters at initialization time.
    """

    def __init__(self, use_mlp=True, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[],
                 num_layers=3, base_channels=64):
        """Initialize the PatchSample module with immediate MLP creation.

        Args:
            use_mlp (bool): Whether to use MLP projection. Default is True.
            init_type (str): Initialization method for weights.
            init_gain (float): Scaling factor for initialization.
            nc (int): Number of channels in MLP hidden layers.
            gpu_ids (list): List of GPU ids to use.
            num_layers (int): Number of expected feature layers.
            base_channels (int): Base channel count for feature estimation.
        """
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

        # Initialize MLPs immediately with estimated input dimensions
        if self.use_mlp:
            self._create_initial_mlps(num_layers, base_channels)
            self.mlp_init = True
        else:
            self.mlp_init = False

    def _create_initial_mlps(self, num_layers, base_channels):
        """Create initial MLPs with estimated dimensions.

        Args:
            num_layers (int): Number of expected feature layers.
            base_channels (int): Base channel count for feature estimation.
        """
        for layer_id in range(num_layers):
            # Estimate feature channels based on layer depth (common in CNNs)
            # Typically channels double with each downsampling layer
            estimated_channels = base_channels * (2 ** min(layer_id, 3))

            # Create MLP for this layer
            mlp = nn.Sequential(
                nn.Linear(estimated_channels, self.nc),
                nn.ReLU(),
                nn.Linear(self.nc, self.nc)
            )

            # Move to GPU if needed
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                mlp.cuda()

            # Save the MLP as a module attribute
            setattr(self, f'mlp_{layer_id}', mlp)

        # Initialize weights
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)

    def _resize_mlp_if_needed(self, feats):
        """Resize MLP input layers if actual feature dimensions don't match.

        Args:
            feats (list): List of feature tensors from the network.
        """
        for feat_id, feat in enumerate(feats):
            input_nc = feat.shape[1]  # Actual feature channels
            mlp = getattr(self, f'mlp_{feat_id}')

            # Check if we need to resize the first linear layer
            current_in_features = mlp[0].in_features

            if current_in_features != input_nc:
                # Create a new first linear layer with correct dimensions
                new_linear = nn.Linear(input_nc, self.nc)

                # Initialize the new layer with same method as the original
                if self.init_type == 'normal':
                    nn.init.normal_(new_linear.weight, 0.0, self.init_gain)
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(new_linear.weight, gain=self.init_gain)
                elif self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(new_linear.weight, a=0, mode='fan_in')
                elif self.init_type == 'orthogonal':
                    nn.init.orthogonal_(new_linear.weight, gain=self.init_gain)

                # Initialize bias if present
                if new_linear.bias is not None:
                    nn.init.constant_(new_linear.bias, 0.0)

                # Move to same device as the current layer
                new_linear = new_linear.to(mlp[0].weight.device)

                # Replace the first layer in the sequential
                new_mlp = nn.Sequential(
                    new_linear,
                    mlp[1],  # ReLU
                    mlp[2]  # Second linear layer
                )

                # Update the module
                setattr(self, f'mlp_{feat_id}', new_mlp)

    def forward(self, feats, num_patches=64, patch_ids=None):
        """Forward function to sample patches and apply MLPs.

        Args:
            feats (list): List of feature tensors.
            num_patches (int): Number of patches to sample.
            patch_ids (list): List of patch indices for sampling. If None, will randomly sample.

        Returns:
            tuple: (list of transformed features, list of patch indices)
        """
        return_ids = []
        return_feats = []

        # Check if we need to resize MLPs based on actual feature dimensions
        if self.use_mlp:
            self._resize_mlp_if_needed(feats)

        # Process each feature map
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)

            # Sample patches
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # Use numpy for random permutation to avoid CUDA issues
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]

                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []

            # Apply MLP if enabled
            if self.use_mlp:
                mlp = getattr(self, f'mlp_{feat_id}')
                x_sample = mlp(x_sample)

            # Store results
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            # Reshape if no patches were sampled
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])

            return_feats.append(x_sample)

        return return_feats, return_ids