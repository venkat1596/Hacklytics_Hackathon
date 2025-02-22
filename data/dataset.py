import os
import hashlib
from typing import Optional, Dict
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import numpy as np
import nibabel as nib
import random
import json
from tqdm import tqdm
import cv2


class MRIAugmentation:
    """
    Enhanced MRI augmentation with dimension preservation and proper ordering.
    """

    def __init__(self, p=0.5, target_shape=(128, 128, 150)):
        self.p = p
        self.target_shape = target_shape  # Expected (H, W, D)

    def random_flip(self, image):
        """Random flip that preserves output dimensions"""
        if random.random() < self.p:
            # Only flip along H or W dimensions (0 or 1) to preserve depth
            axis = random.randint(0, 1)
            return np.flip(image, axis)
        return image

    def random_rotate(self, image):
        """90-degree rotation in H-W plane only to preserve depth"""
        if random.random() < self.p:
            # Only rotate in H-W plane (axes=(0,1)) to preserve depth dimension
            k = random.randint(1, 3)  # Number of 90-degree rotations
            return np.rot90(image, k=k, axes=(0, 1))
        return image

    def random_intensity(self, image):
        """Intensity adjustment that preserves image structure"""
        if random.random() < self.p:
            factor = np.random.uniform(0.9, 1.1)
            return image * factor
        return image

    def ensure_shape(self, image):
        """Ensure the image matches the target shape"""
        if image.shape != self.target_shape:
            # Padding or cropping to match target shape
            result = np.zeros(self.target_shape, dtype=image.dtype)
            # Copy the minimum dimensions
            h = min(image.shape[0], self.target_shape[0])
            w = min(image.shape[1], self.target_shape[1])
            d = min(image.shape[2], self.target_shape[2])
            result[:h, :w, :d] = image[:h, :w, :d]
            return result
        return image

    def __call__(self, image):
        """Apply augmentations while preserving dimensions"""
        image = self.random_flip(image)
        image = self.random_rotate(image)
        image = self.random_intensity(image)
        image = self.ensure_shape(image)
        return image


class MRIDataset(torch.utils.data.Dataset):
    """
    Enhanced MRI dataset with proper tensor dimensioning and efficient statistics handling.
    """

    def __init__(self, source_dir, target_dir, split='train', aug_prob=0.5, stats_file='./dataset_stats.json'):
        self.source_dir = os.path.join(source_dir, f'source_{split}')
        self.target_dir = os.path.join(target_dir, f'target_{split}')
        self.stats_file = stats_file

        # Get file lists
        self.source_files = sorted([f for f in os.listdir(self.source_dir) if f.endswith('.nii.gz')])
        self.target_files = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.nii.gz')])

        # Initialize augmentation with target shape
        self.augmentation = MRIAugmentation(p=aug_prob if split == 'train' else 0)

        # Load or calculate statistics
        self._initialize_statistics()

    def _calculate_dataset_hash(self) -> str:
        """
        Calculate a hash that represents the current state of the dataset.
        This hash is based only on filenames and file sizes, ignoring modification times
        to provide more stable caching behavior.

        The hash considers:
        - Names of all files in both source and target directories
        - Sizes of all files (to detect content changes)

        Returns:
            str: A hexadecimal hash string representing the dataset state
        """
        hasher = hashlib.sha256()

        def update_hash_for_directory(directory, file_list):
            # Sort the file list to ensure consistent ordering
            for filename in sorted(file_list):
                filepath = os.path.join(directory, filename)
                # Only include filename and size in hash
                file_stat = os.stat(filepath)
                hash_string = f"{filename}:{file_stat.st_size}"
                hasher.update(hash_string.encode())

        # Process both directories
        update_hash_for_directory(self.source_dir, self.source_files)
        update_hash_for_directory(self.target_dir, self.target_files)

        return hasher.hexdigest()

    def _calculate_image_statistics(self, file_list: list, directory: str) -> Dict[str, float]:
        """
        Calculate statistics on resized images to ensure consistency
        """
        stats = {'min': float('inf'), 'max': float('-inf')}
        total_voxels = 0
        sum_values = 0
        sum_squared = 0

        for file in tqdm(file_list):
            img = nib.load(os.path.join(directory, file)).get_fdata()

            # Resize each slice before calculating statistics
            resized_slices = []
            for i in range(img.shape[2]):
                slice_2d = img[:, :, i]
                resized_slice = cv2.resize(slice_2d, (128, 128), interpolation=cv2.INTER_AREA)
                resized_slices.append(resized_slice)
            img = np.stack(resized_slices, axis=2)

            stats['min'] = min(stats['min'], float(img.min()))
            stats['max'] = max(stats['max'], float(img.max()))

            n_voxels = img.size
            total_voxels += n_voxels
            sum_values += np.sum(img)
            sum_squared += np.sum(img ** 2)

        mean = sum_values / total_voxels
        variance = (sum_squared / total_voxels) - (mean ** 2)
        std = np.sqrt(variance)

        stats['mean'] = float(mean)
        stats['std'] = float(std)
        return stats

    def _initialize_statistics(self):
        """
        Initialize statistics either by loading from file or calculating them.
        Includes checksum verification to ensure data consistency.
        """
        current_hash = self._calculate_dataset_hash()
        should_calculate = True

        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    saved_stats = json.load(f)

                if saved_stats.get('dataset_hash') == current_hash:
                    print("Loading cached dataset statistics...")
                    self.source_stats = saved_stats['source']
                    self.target_stats = saved_stats['target']
                    should_calculate = False
                else:
                    print("Dataset has changed, recalculating statistics...")
            except (json.JSONDecodeError, KeyError):
                print("Statistics file is corrupted, recalculating...")
        else:
            print("No statistics file found, calculating statistics...")

        if should_calculate:
            print("Calculating dataset statistics...")
            self.source_stats = self._calculate_image_statistics(self.source_files, self.source_dir)
            self.target_stats = self._calculate_image_statistics(self.target_files, self.target_dir)

            # Save statistics with hash
            stats_dict = {
                'dataset_hash': current_hash,
                'source': self.source_stats,
                'target': self.target_stats
            }

            with open(self.stats_file, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            print("Statistics calculation complete and saved to file.")

    def normalize(self, image: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
        """
        Normalize image to [-1, 1] range using pre-computed statistics.
        """
        min_val, max_val = stats['min'], stats['max']
        normalized = (image - min_val) / (max_val - min_val)
        normalized = normalized * 2 - 1
        return normalized

    def _process_image(self, image_path: str, is_source: bool = True) -> tuple:
        """
        Process a single image with proper dimension handling.
        Returns: tensor, min_val, max_val
        """
        img = nib.load(image_path).get_fdata()

        # Store original min/max if target
        orig_min = float(img.min()) if not is_source else None
        orig_max = float(img.max()) if not is_source else None

        # Resize each slice along the depth dimension
        resized_slices = []
        for i in range(img.shape[2]):
            slice_2d = img[:, :, i]
            # Using cv2.INTER_AREA for downsampling to preserve structure
            resized_slice = cv2.resize(slice_2d, (128, 128), interpolation=cv2.INTER_AREA)
            resized_slices.append(resized_slice)

        # Stack the resized slices back together
        img = np.stack(resized_slices, axis=2)

        # Apply augmentation
        img = self.augmentation(img)

        # Normalize
        stats = self.source_stats if is_source else self.target_stats
        img = self.normalize(img, stats)

        # Convert to tensor and rearrange dimensions to (C, D, H, W)
        tensor = torch.FloatTensor(img)
        tensor = tensor.permute(2, 0, 1)  # (H, W, D) -> (D, H, W)
        tensor = tensor.unsqueeze(0)  # (D, H, W) -> (C, D, H, W)

        return tensor, orig_min, orig_max

    def __len__(self):
        return max(len(self.source_files), len(self.target_files))

    def __getitem__(self, idx):
        source_idx = idx % len(self.source_files)
        target_idx = idx % len(self.target_files)

        # Process source image
        source_path = os.path.join(self.source_dir, self.source_files[source_idx])
        source_tensor, _, _ = self._process_image(source_path, is_source=True)

        # Process target image
        target_path = os.path.join(self.target_dir, self.target_files[target_idx])
        target_tensor, target_min, target_max = self._process_image(target_path, is_source=False)

        return {
            'source': source_tensor,  # Shape: (C, D, H, W)
            'target': target_tensor,  # Shape: (C, D, H, W)
            'source_path': source_path,
            'target_path': target_path,
            'target_min': target_min,
            'target_max': target_max,
            'source_global_min': self.source_stats['min'],
            'source_global_max': self.source_stats['max'],
            'target_global_min': self.target_stats['min'],
            'target_global_max': self.target_stats['max'],
            'target_mean': self.target_stats['mean'],
            'target_std': self.target_stats['std']
        }

class MRIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MRI image-to-image translation.
    """

    def __init__(self, source_dir, target_dir, batch_size=4, num_workers=4,
                 aug_prob=0.5, pin_memory=True):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.source_stats = None
        self.target_stats = None

    def prepare_data(self):
        """One-time preparation"""
        # Create temporary dataset to calculate statistics
        temp_train = MRIDataset(
            self.hparams.source_dir,
            self.hparams.target_dir,
            split='train',
            aug_prob=0.0
        )

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage"""
        # Load normalization statistics
        if os.path.exists('normalization_stats.json'):
            with open('normalization_stats.json', 'r') as f:
                stats = json.load(f)
                self.source_stats = stats['source']
                self.target_stats = stats['target']

        if stage == 'fit' or stage is None:
            self.train_dataset = MRIDataset(
                self.hparams.source_dir,
                self.hparams.target_dir,
                split='train',
                aug_prob=self.hparams.aug_prob
            )

            self.val_dataset = MRIDataset(
                self.hparams.source_dir,
                self.hparams.target_dir,
                split='valid',
                aug_prob=0.0
            )

        if stage == 'test' or stage is None:
            self.test_dataset = MRIDataset(
                self.hparams.source_dir,
                self.hparams.target_dir,
                split='test',
                aug_prob=0.0
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def get_normalization_stats(self):
        return {
            'source': self.source_stats,
            'target': self.target_stats
        }