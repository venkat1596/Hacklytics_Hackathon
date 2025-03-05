import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import cv2 as cv
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
        normalized = (normalized - 0.5) / 0.5
        return normalized

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
            'target_global_max': torch.tensor(self.target_stats['max']),
            'image_name': self.source_files[source_idx].name.split('.')[0]
        }


class MRITargetDataset(torch.utils.data.Dataset):
    def __init__(self, source_dir, target_dir, stats_file='./dataset_stats.json'):
        self.img_pairs = self._calculate_pair_file(source_dir, target_dir)
        self.stats_file = stats_file

        if len(self.img_pairs) == 0:
            raise ValueError("No image pairs found. Check source and target directories.")
        self.source_stats, self.target_stats = self._get_or_calculate_stats()

    def __len__(self):
        return len(self.img_pairs)

    def normalize_image(self, img, stats):
        """Normalize image to [-1, 1] range using domain statistics"""
        img_array = np.array(img, dtype=np.float32)
        normalized = (img_array - stats['min']) / (stats['max'] - stats['min'] + 1e-6)
        normalized = (normalized - 0.5) / 0.5
        return normalized

    def __getitem__(self, idx):
        source_path, target_path = self.img_pairs[idx]
        source_img = Image.open(source_path).convert('F')
        source_norm = self.normalize_image(source_img, self.source_stats)

        target_img = Image.open(target_path).convert('F')
        target_norm = self.normalize_image(target_img, self.target_stats)

        return {
            'source': torch.from_numpy(source_norm),
            'target': torch.from_numpy(target_norm),
            'source_global_min': torch.tensor(self.source_stats['min']),
            'source_global_max': torch.tensor(self.source_stats['max']),
            'target_global_min': torch.tensor(self.target_stats['min']),
            'target_global_max': torch.tensor(self.target_stats['max']),
            'image_name': Path(source_path).stem
        }

    def _calculate_pair_file(self, source_dir, target_dir):
        source_dir = Path(source_dir).absolute()
        target_dir = Path(target_dir).absolute()

        supported_ext = ["*.png", "*.jpg", "*.jpeg"]

        source_files = []
        for ext in supported_ext:
            source_files.extend(list(source_dir.rglob(ext)))

        img_pair_list = []

        for img_path in source_files:
            img_name = img_path.stem
            img_parent = img_path.parent
            # Check for multiple possible image extensions
            target_base = target_dir / img_parent / img_name
            possible_extensions = ['.png', '.jpg', '.jpeg']
            target_path = None

            # Try each possible extension
            for ext in possible_extensions:
                possible_path = target_base.with_suffix(ext)
                if possible_path.exists():
                    target_path = possible_path
                    break

            if target_path is None:
                print(f"Target image not found for {img_name} (tried png, jpg, jpeg)")
                continue

            img_pair_list.append((img_path, target_path))

        return img_pair_list

    def _get_or_calculate_stats(self):
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

            max_source_dir = float('-inf')
            min_source_dir = float('inf')

            max_target_dir = float('-inf')
            min_target_dir = float('inf')

            for source_img, target_img in self.img_pairs:
                source_img = Image.open(source_img).convert('F')
                target_img = Image.open(target_img).convert('F')

                source_img = np.array(source_img)
                target_img = np.array(target_img)

                max_source_dir = max(max_source_dir, np.max(source_img))
                min_source_dir = min(min_source_dir, np.min(source_img))

                max_target_dir = max(max_target_dir, np.max(target_img))
                min_target_dir = min(min_target_dir, np.min(target_img))

            source_stats = {'max': float(max_source_dir), 'min': float(min_source_dir)}
            target_stats = {'max': float(max_target_dir), 'min': float(min_target_dir)}

            # Get current timestamp
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save statistics to JSON file
            stats = {
                'source': source_stats,
                'target': target_stats,
                'metadata': {
                    'pair_files_list': len(self.img_pairs),
                    'date_calculated': current_time
                }
            }

            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=4)

            print(f"Statistics saved to {self.stats_file}")
            return source_stats, target_stats


class MRITargetDataModule(pl.LightningDataModule):
    def __init__(self, train_source_dir, train_target_dir, valid_source_dir, valid_target_dir,
                    stats_file='./dataset_stats.json', batch_size=1, num_workers=4):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_dataset = MRITargetDataset(
            self.hparams.train_source_dir,
            self.hparams.train_target_dir,
            self.hparams.stats_file
        )

        self.valid_dataset = MRITargetDataset(
            self.hparams.valid_source_dir,
            self.hparams.valid_target_dir,
            self.hparams.stats_file
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
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )




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