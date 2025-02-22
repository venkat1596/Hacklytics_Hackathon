import os
import nibabel as nib
import numpy as np
import imageio.v2 as imageio
from pathlib import Path

def convert_nii_to_jpg(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    fileNo = 0

    # Loop through all .nii.gz files in the input folder
    for nii_file in Path(input_folder).glob("*.nii.gz"):
        # Load the .nii.gz file
        img = nib.load(str(nii_file))
        data = img.get_fdata()

        # Normalize data to 0-255
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        data = data.astype(np.uint8)

        # Create a subfolder for each .nii.gz file
    #    file_output_folder = os.path.join(output_folder, nii_file.stem)
    #    os.makedirs(file_output_folder, exist_ok=True)
    
        sThird = 0  # Track a value to save every third
        # Save each slice as a .jpg image
        for i in range(data.shape[2]):  # Assuming slices are along the third dimension
            if sThird % 4 == 0:
                slice_img = data[:, :, i]
                output_file = os.path.join(output_folder, f"{fileNo}-{sThird}slice_{i:03d}.jpg")
                imageio.imwrite(output_file, slice_img)
            sThird += 1  # Increment the counter for saving every third slice
        fileNo += 1
        
        
        print(f"Converted {nii_file} to JPG slices in {output_folder}")


convert_nii_to_jpg("T1-organized/TrainA",  "OrgJpg/trainA")
convert_nii_to_jpg("T1-organized/TrainB",  "OrgJpg/trainB")
convert_nii_to_jpg("T1-organized/TestA",   "OrgJpg/testA")
convert_nii_to_jpg("T1-organized/TestB",   "OrgJpg/testB")