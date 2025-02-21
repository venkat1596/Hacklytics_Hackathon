import os
import shutil
from pathlib import Path
from tqdm import tqdm
import nibabel as nib


def setup_directories(base_path):
    """
    Create the necessary directory structure for organizing MRI files.

    Args:
        base_path (str): The base directory where source and target folders will be created

    Returns:
        tuple: Paths to source (1.5T) and target (3T) directories
    """
    # Create paths for 1.5T and 3T images
    source_dir = Path(base_path) / 'source_1.5T'
    target_dir = Path(base_path) / 'target_3T'

    # Create directories if they don't exist
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    return source_dir, target_dir


def identify_scanner_strength(filename):
    """
    Determine the scanner strength based on the hospital identifier in the filename.

    Args:
        filename (str): Name of the NIfTI file

    Returns:
        str: Scanner strength ('1.5T' or '3T')
    """
    filename = filename.upper()

    # HH = Hammersmith Hospital (3T)
    if 'HH' in filename:
        return '3T'
    # GUYS or IOP = Guy's Hospital or Institute of Psychiatry (1.5T)
    elif any(id in filename for id in ['GUYS', 'IOP']):
        return '1.5T'
    else:
        return None


def organize_mri_files(input_dir, base_output_dir):
    """
    Organize MRI files into appropriate directories based on scanner strength.

    Args:
        input_dir (str): Directory containing the input .nii.gz files
        base_output_dir (str): Base directory where sorted files will be stored

    Returns:
        dict: Summary of file movements
    """
    # Setup directory structure
    source_dir, target_dir = setup_directories(base_output_dir)

    # Initialize counters for summary
    summary = {
        '3T': 0,
        '1.5T': 0,
        'unidentified': 0
    }

    # Process each .nii.gz file
    for file in tqdm(Path(input_dir).glob('*.nii.gz')):
        # Determine scanner strength
        scanner_strength = identify_scanner_strength(file.name)

        if scanner_strength == '3T':
            destination = target_dir / file.name
            summary['3T'] += 1
        elif scanner_strength == '1.5T':
            destination = source_dir / file.name
            summary['1.5T'] += 1
        else:
            print(f"Warning: Could not identify scanner strength for {file.name}")
            summary['unidentified'] += 1
            continue

        # Copy file to appropriate directory
        shutil.copy2(file, destination)
        print(f"Moved {file.name} to {scanner_strength} directory")

    return summary


def main():
    # Set your input and output directories
    input_directory = "./tar_files/IXI-T1"
    output_base_directory = "./T1-organized"

    # Process the files
    summary = organize_mri_files(input_directory, output_base_directory)

    # Print summary
    print("\nProcessing Summary:")
    print(f"3T files processed: {summary['3T']}")
    print(f"1.5T files processed: {summary['1.5T']}")
    if summary['unidentified'] > 0:
        print(f"Unidentified files: {summary['unidentified']}")


if __name__ == "__main__":
    main()