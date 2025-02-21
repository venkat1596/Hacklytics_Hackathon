import numpy as np
from pathlib import Path
from scipy.ndimage import rotate, shift
import nibabel as nib
import json


class MRIContrastiveAugmenter:
    def __init__(self, save_dir):
        """
        Initialize augmenter with directory to save augmented images

        Parameters:
            save_dir (str): Directory to save augmented images and metadata
        """
        self.save_dir = Path(save_dir)
        self.source_dir = self.save_dir / 'source_augmented'
        self.target_dir = self.save_dir / 'target_augmented'

        # Create directories if they don't exist
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Keep track of augmentation parameters for reproducibility
        self.augmentation_metadata = {}

    def basic_augment(self, image, augment_params):
        """
        Apply basic augmentations suitable for both source and target

        Parameters:
            image: 3D numpy array
            augment_params: Dictionary containing augmentation parameters
        """
        augmented = image.copy()

        # Rotation - small angles to maintain anatomy
        if augment_params.get('rotate'):
            angles = np.random.uniform(-10, 10, size=3)
            for axis in range(3):
                augmented = rotate(augmented, angles[axis],
                                   axes=((axis + 1) % 3, (axis + 2) % 3),
                                   mode='reflect')
            augment_params['rotation_angles'] = angles.tolist()

        # Flip along valid anatomical axes
        if augment_params.get('flip'):
            # Only flip left-right (anatomically valid)
            if np.random.random() > 0.5:
                augmented = np.flip(augmented, axis=0)
                augment_params['flipped_axis'] = 0

        # Small translations
        if augment_params.get('translate'):
            shifts = np.random.randint(-10, 10, size=3)
            augmented = shift(augmented, shifts, mode='reflect')
            augment_params['shifts'] = shifts.tolist()

        return augmented, augment_params

    def advanced_augment(self, image, augment_params):
        """
        Apply additional augmentations suitable only for source images

        Parameters:
            image: 3D numpy array
            augment_params: Dictionary containing augmentation parameters
        """
        augmented = image.copy()

        # Add Rician noise (common in MRI)
        if augment_params.get('noise'):
            std = np.random.uniform(0.01, 0.03)
            noise_real = np.random.normal(0, std, image.shape)
            noise_imag = np.random.normal(0, std, image.shape)
            augmented = np.sqrt((augmented + noise_real) ** 2 + noise_imag ** 2)
            augment_params['noise_std'] = float(std)

        # Intensity variations
        if augment_params.get('intensity'):
            gamma = np.random.uniform(0.8, 1.2)
            augmented = np.power(augmented, gamma)
            augment_params['gamma'] = float(gamma)

        return augmented, augment_params

    def generate_augmentations(self, source_image, target_image,
                               base_filename, num_augmentations=40):
        """
        Generate multiple augmentations for a pair of source and target images

        Parameters:
            source_image: 3D numpy array of source (1.5T) image
            target_image: 3D numpy array of target (3T) image
            base_filename: Original filename without extension
            num_augmentations: Number of augmented pairs to generate
        """
        for i in range(num_augmentations):
            # Initialize augmentation parameters
            augment_params = {
                'rotate': True,
                'flip': True,
                'translate': True,
                'noise': True,
                'intensity': True,
                'augmentation_id': i
            }

            # Apply basic augmentations to both source and target
            # Use same parameters for both to maintain correspondence
            augmented_source, params_source = self.basic_augment(
                source_image, augment_params.copy())
            augmented_target, _ = self.basic_augment(
                target_image, augment_params.copy())

            # Apply additional augmentations only to source
            augmented_source, params_source = self.advanced_augment(
                augmented_source, params_source)

            # Save augmented images
            aug_filename = f"{base_filename}_aug_{i:03d}.nii.gz"

            # Save source
            nib.save(nib.Nifti1Image(augmented_source, np.eye(4)),
                     self.source_dir / aug_filename)

            # Save target
            nib.save(nib.Nifti1Image(augmented_target, np.eye(4)),
                     self.target_dir / aug_filename)

            # Store metadata
            self.augmentation_metadata[aug_filename] = params_source

        # Save augmentation metadata
        with open(self.save_dir / 'augmentation_metadata.json', 'w') as f:
            json.dump(self.augmentation_metadata, f, indent=4)


def main():
    # Example usage
    input_source = "path/to/source/image.nii.gz"
    input_target = "path/to/target/image.nii.gz"
    save_dir = "path/to/save/augmentations"

    # Load images
    source_img = nib.load(input_source).get_fdata()
    target_img = nib.load(input_target).get_fdata()

    # Create augmenter
    augmenter = MRIContrastiveAugmenter(save_dir)

    # Generate augmentations
    augmenter.generate_augmentations(
        source_img,
        target_img,
        Path(input_source).stem,
        num_augmentations=40
    )


if __name__ == "__main__":
    main()