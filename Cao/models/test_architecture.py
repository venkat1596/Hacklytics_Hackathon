import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import gc
from pathlib import Path
from PIL import Image

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

def load_slices(base_path, type_folder):
    """Load all slices from a folder"""
    folder_path = Path(base_path) / type_folder
    slices = []
    
    # Get all image files and sort them
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for file_name in files:
        img_path = folder_path / file_name
        # Load image in grayscale
        img = Image.open(img_path).convert('L')
        # Resize to 128x128
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        slices.append(img_array)
    
    return np.stack(slices)

def preprocess_volumes(base_path):
    """Load and preprocess both 1.5T and 3T volumes"""
    # Load slices from both folders
    low_field_volume = load_slices(base_path, '1.5T')
    high_field_volume = load_slices(base_path, '3T')
    
    print(f"Loaded volumes shapes - 1.5T: {low_field_volume.shape}, 3T: {high_field_volume.shape}")
    return low_field_volume, high_field_volume

def visualize_sample_slices(input_vol, generated_vol, target_vol=None, num_samples=3):
    """Visualize sample slices from the volumes"""
    total_slices = input_vol.shape[0]
    # Choose evenly spaced slice indices
    indices = np.linspace(10, total_slices-10, num_samples, dtype=int)
    
    if target_vol is not None:
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 8))
        plt.suptitle('Sample Slices Comparison')
    else:
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        plt.suptitle('Input vs Generated Slices')
    
    for i, idx in enumerate(indices):
        # Input slice
        axes[0, i].imshow(input_vol[idx], cmap='gray')
        axes[0, i].set_title(f'1.5T Slice {idx}')
        axes[0, i].axis('off')
        
        # Generated slice
        axes[1, i].imshow(generated_vol[idx], cmap='gray')
        axes[1, i].set_title(f'Generated 3T Slice {idx}')
        axes[1, i].axis('off')
        
        # Target slice (if provided)
        if target_vol is not None:
            axes[2, i].imshow(target_vol[idx], cmap='gray')
            axes[2, i].set_title(f'Target 3T Slice {idx}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_model():
    print("Starting test with multiple slices...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load data
        base_path = "."  # Current directory
        low_field_vol, high_field_vol = preprocess_volumes(base_path)
        
        # Convert to torch tensors
        low_field_tensor = torch.from_numpy(low_field_vol).unsqueeze(1).to(device)  # [150, 1, 128, 128]
        high_field_tensor = torch.from_numpy(high_field_vol).unsqueeze(1).to(device)
        
        # Initialize model
        model = MRICycleFreeGAN(device=device)
        
        print("\nTesting conversion...")
        with torch.no_grad():
            # Process in smaller batches to save memory
            batch_size = 10
            generated_slices = []
            
            for i in range(0, low_field_tensor.size(0), batch_size):
                batch = low_field_tensor[i:i+batch_size]
                generated_batch = model.generator(batch)
                generated_slices.append(generated_batch.cpu())
            
            generated_vol = torch.cat(generated_slices, dim=0)
        
        # Convert back to numpy for visualization
        input_vol = low_field_tensor.cpu().squeeze().numpy()
        generated_vol = generated_vol.squeeze().numpy()
        target_vol = high_field_tensor.cpu().squeeze().numpy()
        
        # Visualize sample slices
        print("\nVisualizing sample slices...")
        visualize_sample_slices(input_vol, generated_vol, target_vol, num_samples=3)
        
        print("Test completed successfully!")
        return model
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = test_model()