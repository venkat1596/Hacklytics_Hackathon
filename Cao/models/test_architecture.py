import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Import all the model classes defined earlier
from mri_cycle_free_gan import (
    SpectralNormConv2d,
    MRIInvertibleBlock,
    MRIInvertibleGenerator,
    MRIDiscriminator,
    MRICycleFreeGAN
)

def test_architecture():
    print("Starting architecture test...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Initialize model
        print("Initializing model...")
        model = MRICycleFreeGAN(device=device)
        
        # Create a test batch (simulating 1.5T MRI)
        print("Creating test data...")
        batch_size = 2
        test_images = torch.randn(batch_size, 1, 256, 256).to(device)
        
        print("\nTesting individual components:")
        
        # Test generator forward pass
        print("1. Testing generator forward pass...")
        fake_3t = model.generator(test_images)
        print(f"   Output shape: {fake_3t.shape}")
        
        # Test generator inverse pass
        print("2. Testing generator inverse pass...")
        reconstructed = model.generator.inverse(fake_3t)
        print(f"   Reconstruction shape: {reconstructed.shape}")
        
        # Test discriminator
        print("3. Testing discriminator...")
        disc_output = model.discriminator(fake_3t)
        print(f"   Discriminator output shape: {disc_output.shape}")
        
        # Test full training step
        print("\n4. Testing training step...")
        # Create fake "high field" images for testing
        fake_high_field = torch.randn_like(test_images)
        losses = model.train_step(test_images, fake_high_field)
        print("   Training step completed successfully")
        print("   Losses:", losses)
        
        # Visualize results
        print("\nGenerating visualizations...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Architecture Test Results')
        
        # First row - first image
        axes[0,0].imshow(test_images[0, 0].cpu().detach().numpy(), cmap='gray')
        axes[0,0].set_title('Input (1.5T-like)')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(fake_3t[0, 0].cpu().detach().numpy(), cmap='gray')
        axes[0,1].set_title('Generated (3T-like)')
        axes[0,1].axis('off')
        
        axes[0,2].imshow(reconstructed[0, 0].cpu().detach().numpy(), cmap='gray')
        axes[0,2].set_title('Reconstructed (1.5T-like)')
        axes[0,2].axis('off')
        
        # Second row - second image
        axes[1,0].imshow(test_images[1, 0].cpu().detach().numpy(), cmap='gray')
        axes[1,0].set_title('Input (1.5T-like)')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(fake_3t[1, 0].cpu().detach().numpy(), cmap='gray')
        axes[1,1].set_title('Generated (3T-like)')
        axes[1,1].axis('off')
        
        axes[1,2].imshow(reconstructed[1, 0].cpu().detach().numpy(), cmap='gray')
        axes[1,2].set_title('Reconstructed (1.5T-like)')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and print reconstruction error
        recon_error = F.mse_loss(test_images, reconstructed)
        print(f"\nReconstruction error: {recon_error.item():.6f}")
        
        print("\nArchitecture test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_architecture()