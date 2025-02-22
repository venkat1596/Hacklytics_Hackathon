import os
import shutil
import random
from pathlib import Path


def split_unpaired_datasets(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split unpaired source (1.5T) and target (3T) image datasets for image-to-image translation.
    Each domain is split independently since we don't need paired images.

    This is suitable for unpaired translation approaches like CycleGAN where we want to learn
    the mapping between domains without having exact pairs.

    Args:
        source_dir (str): Path to source (1.5T) images directory
        target_dir (str): Path to target (3T) images directory
        train_ratio (float): Ratio of training data (default: 0.7)
        val_ratio (float): Ratio of validation data (default: 0.15)
        test_ratio (float): Ratio of test data (default: 0.15)
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)

    def split_single_domain(input_dir, domain_name):
        """Helper function to split a single domain's dataset"""
        # Get all .nii.gz files
        files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii.gz')])
        n_samples = len(files)

        # Calculate split sizes
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val

        # Create random split indices
        indices = list(range(n_samples))
        random.shuffle(indices)

        splits = {
            'train': indices[:n_train],
            'valid': indices[n_train:n_train + n_val],
            'test': indices[n_train + n_val:]
        }

        source_output_dir = Path("./data/source-dir")
        target_output_dir = Path("./data/target-dir")
        source_output_dir.mkdir(parents=True, exist_ok=True)
        target_output_dir.mkdir(parents=True, exist_ok=True)

        # Create directories and copy files
        for split_name, split_indices in splits.items():
            # Create output directory
            if domain_name == 'source':
                output_dir = source_dir / f'{domain_name}_{split_name}'
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = target_dir / f'{domain_name}_{split_name}'
                output_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for idx in split_indices:
                src_file = files[idx]
                shutil.copy2(
                    os.path.join(input_dir, src_file),
                    os.path.join(output_dir, src_file)
                )

        return {
            'total': n_samples,
            'train': n_train,
            'valid': n_val,
            'test': n_test
        }



    # Split both domains
    print("Processing source domain (1.5T)...")
    source_stats = split_single_domain(source_dir, 'source')

    print("Processing target domain (3T)...")
    target_stats = split_single_domain(target_dir, 'target')




    # Print statistics
    print("\nDataset Statistics:")
    print("\nSource Domain (1.5T):")
    print(f"Total samples: {source_stats['total']}")
    print(f"Training samples: {source_stats['train']}")
    print(f"Validation samples: {source_stats['valid']}")
    print(f"Test samples: {source_stats['test']}")

    print("\nTarget Domain (3T):")
    print(f"Total samples: {target_stats['total']}")
    print(f"Training samples: {target_stats['train']}")
    print(f"Validation samples: {target_stats['valid']}")
    print(f"Test samples: {target_stats['test']}")

    print("\nDirectory structure created:")
    print("source_train/  - Source domain training images")
    print("source_valid/  - Source domain validation images")
    print("source_test/   - Source domain test images")
    print("target_train/  - Target domain training images")
    print("target_valid/  - Target domain validation images")
    print("target_test/   - Target domain test images")


if __name__ == "__main__":
    # Example usage
    split_unpaired_datasets(
        source_dir='./data/T1-organized/source_1.5T',
        target_dir='./data/T1-organized/target_3T',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )