import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import json
import random


class UnpairedMRIDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform_source=None, transform_target=None,
                 stats_file='dataset_stats.json'):
        """
        Args:
            source_dir (str): Directory with source images
            target_dir (str): Directory with target images
            transform_source (callable, optional): Transforms for source images
            transform_target (callable, optional): Transforms for target images
            stats_file (str): Path to JSON file storing dataset statistics
        """
        self.source_files = sorted(list(Path(source_dir).rglob('*.jpg')))
        self.target_files = sorted(list(Path(target_dir).rglob('*.jpg')))

        # Get or calculate statistics
        self.stats_file = stats_file
        self.source_stats, self.target_stats = self._get_or_calculate_stats(
            source_dir, target_dir
        )

        self.transform_source = transform_source
        self.transform_target = transform_target

    def _calculate_stats(self, files, domain_name):
        """Calculate global statistics for normalization"""
        print(f"Calculating statistics for {domain_name} domain...")
        max_val = float('-inf')
        min_val = float('inf')

        for file in files:
            img = Image.open(file).convert('L')
            img_array = np.array(img)
            max_val = max(max_val, np.max(img_array))
            min_val = min(min_val, np.min(img_array))

        return {'max': float(max_val), 'min': float(min_val)}

    def _get_or_calculate_stats(self, source_dir, target_dir):
        """Get statistics from JSON file or calculate if not exists"""
        # Create stats directory if it doesn't exist
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)

        if os.path.exists(self.stats_file):
            print(f"Loading statistics from {self.stats_file}")
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
                return stats['source'], stats['target']
        else:
            print("Statistics file not found. Calculating statistics...")
            source_stats = self._calculate_stats(self.source_files, "source")
            target_stats = self._calculate_stats(self.target_files, "target")

            # Get current timestamp
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save statistics to JSON file
            stats = {
                'source': source_stats,
                'target': target_stats,
                'metadata': {
                    'source_dir': str(source_dir),
                    'target_dir': str(target_dir),
                    'source_files': len(self.source_files),
                    'target_files': len(self.target_files),
                    'date_calculated': current_time
                }
            }

            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=4)

            print(f"Statistics saved to {self.stats_file}")
            return source_stats, target_stats

    def normalize_image(self, img, stats):
        """Normalize image to [-1, 1] range using domain statistics"""
        img_array = np.array(img, dtype=np.float32)
        normalized = (img_array - stats['min']) / (stats['max'] - stats['min'] + 1e-6)
        return (normalized * 2) - 1

    def __len__(self):
        return max(len(self.source_files), len(self.target_files))

    def __getitem__(self, idx):
        # Handle different dataset sizes
        source_idx = idx % len(self.source_files)
        target_idx = idx % len(self.target_files)

        # Load and normalize source image
        source_img = Image.open(self.source_files[source_idx]).convert('L')
        source_img = self.normalize_image(source_img, self.source_stats)
        source_tensor = torch.from_numpy(source_img).unsqueeze(0)

        # Load and normalize target image
        target_img = Image.open(self.target_files[target_idx]).convert('L')
        target_img = self.normalize_image(target_img, self.target_stats)
        target_tensor = torch.from_numpy(target_img).unsqueeze(0)

        # Apply transforms
        if self.transform_source is not None:
            source_tensor = self.transform_source(source_tensor)
        if self.transform_target is not None:
            target_tensor = self.transform_target(target_tensor)

        return {
            'source': source_tensor,
            'target': target_tensor,
            'source_global_min': torch.tensor(self.source_stats['min']),
            'source_global_max': torch.tensor(self.source_stats['max']),
            'target_global_min': torch.tensor(self.target_stats['min']),
            'target_global_max': torch.tensor(self.target_stats['max'])
        }


class MRIDataModule2D(pl.LightningDataModule):
    def __init__(
            self,
            train_source_dir: str,
            train_target_dir: str,
            valid_source_dir: str,
            valid_target_dir: str,
            stats_file: str = 'dataset_stats.json',
            batch_size: int = 1,
            num_workers: int = 4,
            source_rotation_angle: float = 10.0,
            source_translation: float = 0.1,
            target_rotation_angle: float = 5.0,
            target_translation: float = 0.05
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create transforms for source images (more aggressive)
        self.source_transforms = transforms.Compose([
            transforms.RandomRotation(source_rotation_angle),
            transforms.RandomAffine(
                degrees=0,
                translate=(source_translation, source_translation)
            ),
        ])

        # Create transforms for target images (more conservative)
        self.target_transforms = transforms.Compose([
            transforms.RandomRotation(target_rotation_angle),
            transforms.RandomAffine(
                degrees=0,
                translate=(target_translation, target_translation)
            ),
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = UnpairedMRIDataset(
                self.hparams.train_source_dir,
                self.hparams.train_target_dir,
                transform_source=self.source_transforms,
                transform_target=self.target_transforms,
                stats_file=self.hparams.stats_file
            )

            # Create validation dataset with no augmentations
            self.val_dataset = UnpairedMRIDataset(
                self.hparams.valid_source_dir,
                self.hparams.valid_target_dir,
                stats_file=self.hparams.stats_file
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )