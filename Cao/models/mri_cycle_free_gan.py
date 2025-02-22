import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralNormConv2d(nn.Module):
    """Convolution layer with spectral normalization for stability"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        return self.conv(x)

class MRIInvertibleGenerator(nn.Module):
    def __init__(self, in_channels=1, num_blocks=4):
        super().__init__()
        
        # Reduced number of features for memory efficiency
        self.initial_conv = nn.Sequential(
            SpectralNormConv2d(in_channels, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(16, 32, 3, 1, 1)
        )
        
        # Main processing blocks with reduced channels
        self.blocks = nn.ModuleList([MRIInvertibleBlock(32) for _ in range(num_blocks)])
        
        # Contrast modulation with reduced channels
        self.contrast_mod = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 1),
            nn.Sigmoid()
        )
        
        self.final_conv = nn.Sequential(
            SpectralNormConv2d(32, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(16, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        features = self.initial_conv(x)
        
        for block in self.blocks:
            features = block(features)
            scale = self.contrast_mod(features)
            features = features * scale
            
        return self.final_conv(features)

    def inverse(self, y):
        features = self.initial_conv(y)
        
        for block in reversed(self.blocks):
            scale = self.contrast_mod(features)
            features = features / (scale + 1e-6)
            features = block.inverse(features)
            
        return self.final_conv(features)

class MRIInvertibleGenerator(nn.Module):
    def __init__(self, in_channels=1, num_blocks=6):
        super().__init__()
        self.initial_conv = SpectralNormConv2d(in_channels, 64, 3, 1, 1)
        self.blocks = nn.ModuleList([MRIInvertibleBlock(64) for _ in range(num_blocks)])
        self.final_conv = SpectralNormConv2d(64, in_channels, 3, 1, 1)
        
        # Contrast adaptation layers
        self.contrast_modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Initial feature extraction
        features = self.initial_conv(x)
        
        # Process through invertible blocks
        for block in self.blocks:
            features = block(features)
            # Apply contrast modulation
            scale = self.contrast_modulation(features)
            features = features * scale
            
        # Final reconstruction
        return self.final_conv(features)
    
    def inverse(self, y):
        # Initial feature extraction
        features = self.initial_conv(y)
        
        # Process through blocks in reverse order
        for block in reversed(self.blocks):
            # Inverse contrast modulation
            scale = self.contrast_modulation(features)
            features = features / (scale + 1e-6)  # Add epsilon for numerical stability
            features = block.inverse(features)
            
        # Final reconstruction
        return self.final_conv(features)

class MRIDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            # Layer 1: Input layer
            SpectralNormConv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            SpectralNormConv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            SpectralNormConv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            SpectralNormConv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            SpectralNormConv2d(512, 1, 4, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.layers(x)

class MRICycleFreeGAN:
    def __init__(self, device='cuda'):
        self.device = device
        self.generator = MRIInvertibleGenerator().to(device)
        self.discriminator = MRIDiscriminator().to(device)
        
        # Initialize optimizers with custom beta values for MRI
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=2e-4, 
            betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=2e-4, 
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.adversarial_loss = nn.MSELoss()  # LSGAN loss
        self.content_loss = nn.L1Loss()
        
    def train_step(self, low_field, high_field):
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        fake_high = self.generator(low_field)
        d_real = self.discriminator(high_field)
        d_fake = self.discriminator(fake_high.detach())
        
        d_loss_real = self.adversarial_loss(d_real, torch.ones_like(d_real))
        d_loss_fake = self.adversarial_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Adversarial loss
        g_fake = self.discriminator(fake_high)
        g_loss_adv = self.adversarial_loss(g_fake, torch.ones_like(g_fake))
        
        # Content loss
        g_loss_content = self.content_loss(fake_high, high_field)
        
        # Cycle consistency through invertibility
        reconstructed_low = self.generator.inverse(fake_high)
        y_loss = self.content_loss(reconstructed_low, low_field)
        
        # Combined loss
        g_loss = g_loss_adv + 10.0 * g_loss_content + 5.0 * y_loss
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'content_loss': g_loss_content.item(),
            'cycle_loss': y_loss.item()
        }
        
    def convert_to_3T(self, low_field_img):
        self.generator.eval()
        with torch.no_grad():
            return self.generator(low_field_img)
            
    def save(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])