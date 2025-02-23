import os
import glob
import torch
import nibabel as nib
import torchvision.transforms as transforms
from tqdm import tqdm


def process_nifti_files(input_dir, output_dir, start_slice=70, end_slice=110):
    """
    Process multiple NIfTI files and save specified slices as JPG images using GPU acceleration.

    Args:
        input_dir (str): Directory containing .nii.gz files
        output_dir (str): Directory to save output JPG files
        start_slice (int): Starting slice number (default: 70)
        end_slice (int): Ending slice number (default: 110)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get list of all .nii.gz files
    nifti_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))

    # Set up image transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda x: x.convert('RGB')),
    ])

    for nifti_path in tqdm(nifti_files, desc="Processing files"):
        try:
            # Load NIfTI file
            nifti_img = nib.load(nifti_path)

            # Get base filename without extension
            base_name = os.path.splitext(os.path.splitext(os.path.basename(nifti_path))[0])[0]

            # Convert to numpy array and verify dimensions
            data = nifti_img.get_fdata()

            # Normalize to 0-255 range
            data = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

            # Convert to torch tensor and move to GPU
            data_tensor = torch.from_numpy(data).to(device)

            # Process specified slices
            for slice_num in range(start_slice, end_slice + 1):
                # Extract slice and transpose to get 256x256
                slice_data = data_tensor[:, :, slice_num]

                # Normalize slice if needed (ensure proper contrast)
                slice_min = slice_data.min()
                slice_max = slice_data.max()
                if slice_max > slice_min:  # Avoid division by zero
                    slice_data = ((slice_data - slice_min) / (slice_max - slice_min) * 255).to(torch.uint8)

                # Move back to CPU for PIL processing
                slice_data = slice_data.cpu()

                # Convert to PIL Image
                slice_img = transform(slice_data)

                # Save as JPG
                output_path = os.path.join(output_dir, f"{base_name}_slice_{slice_num:03d}.jpg")
                slice_img.save(output_path, quality=95)

        except Exception as e:
            print(f"Error processing {nifti_path}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    input_directory = "./target-dir/target_valid"  # Replace with your input directory
    output_directory = "./Jpegs/3_Valid"  # Replace with your output directory

    process_nifti_files(input_directory, output_directory)