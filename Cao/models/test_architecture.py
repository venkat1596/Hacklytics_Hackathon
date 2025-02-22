import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import gc

# Import all the model classes defined earlier
from mri_cycle_free_gan import (
    SpectralNormConv2d,
    MRIInvertibleBlock,
    MRIInvertibleGenerator,
    MRIDiscriminator,
    MRICycleFreeGAN
)

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_and_preprocess_nii(file_path):
    """Load and preprocess NIfTI file"""
    # Load NIfTI file
    nii_img = nib.load(file_path)
    img_data = nii_img.get_fdata()
    
    # Get middle slice if 3D
    if len(img_data.shape) == 3:
        middle_slice = img_data.shape[2] // 2
        img_data = img_data[:, :, middle_slice]
    
    # Normalize to [0,1]
    p1, p99 = np.percentile(img_data, (1, 99))
    img_data = np.clip(img_data, p1, p99)
    img_data = (img_data - p1) / (p99 - p1)
    
    # Resize to 128x128 if needed
    if img_data.shape != (128, 128):
        img_data = torch.nn.functional.interpolate(
            torch.from_numpy(img_data).float().unsqueeze(0).unsqueeze(0),
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
    
    return img_data

def test_with_real_images():
    print("Starting test with real MRI images...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load images
        print("Loading MRI images...")
        
        low_field_path = "IXI425-IOP-0988-SAGFSPGR_-sIXI42_-0003-00001-000001-01.nii"
        high_field_path = "IXI519-HH-2240-MADisoTFE1_-s3T219_-0301-00003-000001-01.nii"

        low_field = load_and_preprocess_nii(low_field_path)
        high_field = load_and_preprocess_nii(high_field_path)
        
        # Convert to torch tensors
        low_field_tensor = torch.from_numpy(low_field).float().unsqueeze(0).unsqueeze(0).to(device)
        high_field_tensor = torch.from_numpy(high_field).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Initialize model
        print("Initializing model...")
        model = MRICycleFreeGAN(device=device)
        
        print("\nTesting model components:")
        
        # Test generator forward pass (1.5T -> 3T)
        print("1. Testing 1.5T to 3T conversion...")
        with torch.no_grad():
            generated_3t = model.generator(low_field_tensor)
        print(f"   Output shape: {generated_3t.shape}")
        
        clear_gpu_memory()
        
        # Test generator inverse (3T -> 1.5T)
        print("2. Testing 3T to 1.5T conversion...")
        with torch.no_grad():
            generated_1_5t = model.generator.inverse(high_field_tensor)
        print(f"   Output shape: {generated_1_5t.shape}")
        
        clear_gpu_memory()
        
        # Move tensors to CPU for visualization
        low_field_tensor = low_field_tensor.cpu()
        high_field_tensor = high_field_tensor.cpu()
        generated_3t = generated_3t.cpu()
        generated_1_5t = generated_1_5t.cpu()
        
        # Visualize results
        print("\nGenerating visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('MRI Conversion Results')
        
        # First row: 1.5T -> 3T
        axes[0,0].imshow(low_field_tensor[0,0].numpy(), cmap='gray')
        axes[0,0].set_title('Input (1.5T)')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(generated_3t[0,0].detach().numpy(), cmap='gray')
        axes[0,1].set_title('Generated (3T)')
        axes[0,1].axis('off')
        
        # Second row: 3T -> 1.5T
        axes[1,0].imshow(high_field_tensor[0,0].numpy(), cmap='gray')
        axes[1,0].set_title('Input (3T)')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(generated_1_5t[0,0].detach().numpy(), cmap='gray')
        axes[1,1].set_title('Generated (1.5T)')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_gpu_memory()

if __name__ == "__main__":
    test_with_real_images()