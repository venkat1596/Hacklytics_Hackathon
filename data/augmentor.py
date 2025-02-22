

import torch
import torch.nn.functional as F
from pathlib import Path
import nibabel as nib
import json
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.ndimage import rotate, shift
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        for i in tqdm(range(num_augmentations)):
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


@dataclass
class AugmentationStats:
    """Statistics for augmentation quality assessment"""
    mean: float
    std: float
    min_val: float
    max_val: float
    histogram_bins: List[int]
    histogram_values: List[float]


class MRIAugmentationTester:
    """Class to test and validate augmented images"""

    def __init__(self, source_dir: Path, target_dir: Path):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)

    def compute_image_stats(self, image: np.ndarray) -> AugmentationStats:
        """Compute statistical measures for an image"""
        hist_values, hist_bins = np.histogram(image, bins=50, density=True)
        return AugmentationStats(
            mean=float(np.mean(image)),
            std=float(np.std(image)),
            min_val=float(np.min(image)),
            max_val=float(np.max(image)),
            histogram_bins=hist_bins.tolist(),
            histogram_values=hist_values.tolist()
        )

    def validate_augmentation_params(self, metadata: Dict) -> List[str]:
        """Validate augmentation parameters are within expected ranges"""
        issues = []

        for filename, params in metadata.items():
            # Check rotation angles
            if 'rotation_angles' in params:
                if any(abs(angle) > 10 for angle in params['rotation_angles']):
                    issues.append(f"{filename}: Rotation angles exceed ±10 degrees")

            # Check translation
            if 'shifts' in params:
                if any(abs(shift) > 10 for shift in params['shifts']):
                    issues.append(f"{filename}: Translations exceed ±10 pixels")

            # Check noise levels
            if 'noise_std' in params:
                if params['noise_std'] > 0.03:
                    issues.append(f"{filename}: Noise level exceeds 0.03")

            # Check intensity scaling
            if 'gamma' in params:
                if not 0.8 <= params['gamma'] <= 1.2:
                    issues.append(f"{filename}: Gamma value outside [0.8, 1.2]")

        return issues

    def create_visualization(self, original: np.ndarray,
                             augmented: np.ndarray,
                             slice_idx: Optional[int] = None,
                             save_path: Optional[Path] = None) -> None:
        """
        Create interactive visualization comparing original and augmented images
        using Plotly. Saves the plot as an HTML file if save_path is provided.

        Parameters:
            original: Original 3D image array
            augmented: Augmented 3D image array
            slice_idx: Index of slice to visualize (default: middle slice)
            save_path: Path to save the HTML plot (optional)
        """
        if slice_idx is None:
            slice_idx = original.shape[2] // 2

        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original Image', 'Augmented Image'),
            horizontal_spacing=0.05
        )

        # Add original image
        fig.add_trace(
            go.Heatmap(
                z=original[:, :, slice_idx],
                colorscale='Gray',
                showscale=False
            ),
            row=1, col=1
        )

        # Add augmented image
        fig.add_trace(
            go.Heatmap(
                z=augmented[:, :, slice_idx],
                colorscale='Gray',
                showscale=False
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f"Slice Comparison (Slice {slice_idx})",
            width=1000,
            height=500,
            showlegend=False
        )

        # Make axes equal and remove labels
        fig.update_xaxes(showticklabels=False, scaleanchor="y")
        fig.update_yaxes(showticklabels=False)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Visualization saved to {save_path}")

        return fig

    def create_statistics_plot(self, original_stats: AugmentationStats,
                               augmented_stats: AugmentationStats,
                               title: str = "Image Statistics Comparison",
                               save_path: Optional[Path] = None) -> None:
        """
        Create interactive plot comparing statistical distributions
        of original and augmented images.

        Parameters:
            original_stats: Statistics for original image
            augmented_stats: Statistics for augmented image
            title: Plot title
            save_path: Path to save the HTML plot (optional)
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Intensity Histogram', 'Basic Statistics'),
            specs=[[{"type": "xy"}, {"type": "domain"}]]
        )

        # Add histograms
        fig.add_trace(
            go.Scatter(
                x=original_stats.histogram_bins[:-1],
                y=original_stats.histogram_values,
                name='Original',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=augmented_stats.histogram_bins[:-1],
                y=augmented_stats.histogram_values,
                name='Augmented',
                line=dict(color='red')
            ),
            row=1, col=1
        )

        # Add statistics table
        stats_table = go.Table(
            header=dict(
                values=['Metric', 'Original', 'Augmented'],
                align='left'
            ),
            cells=dict(
                values=[
                    ['Mean', 'Std Dev', 'Min', 'Max'],
                    [
                        f"{original_stats.mean:.2f}",
                        f"{original_stats.std:.2f}",
                        f"{original_stats.min_val:.2f}",
                        f"{original_stats.max_val:.2f}"
                    ],
                    [
                        f"{augmented_stats.mean:.2f}",
                        f"{augmented_stats.std:.2f}",
                        f"{augmented_stats.min_val:.2f}",
                        f"{augmented_stats.max_val:.2f}"
                    ]
                ],
                align='left'
            )
        )
        fig.add_trace(stats_table, row=1, col=2)

        # Update layout
        fig.update_layout(
            title_text=title,
            width=1200,
            height=500,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Statistics plot saved to {save_path}")

        return fig





class GPUMRIContrastiveAugmenter:
    def __init__(self, save_dir: str, device: Optional[str] = None):
        """
        Initialize augmenter with GPU support

        Parameters:
            save_dir: Directory to save augmented images
            device: PyTorch device ('cuda' or 'cpu'). If None, automatically selects GPU if available
        """
        self.save_dir = Path(save_dir)
        self.source_dir = self.save_dir / 'source_augmented'
        self.target_dir = self.save_dir / 'target_augmented'

        # Create directories
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Track augmentation parameters
        self.augmentation_metadata = {}

    def rotate3d(self, volume: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Rotate 3D volume along all three axes using PyTorch

        Parameters:
            volume: 4D tensor (1, 1, D, H, W)
            angles: Tensor of rotation angles in degrees (3,)
        """
        # Convert angles to radians
        angles = angles * torch.pi / 180.0

        # Create rotation matrices for each axis
        for axis, angle in enumerate(angles):
            cos = torch.cos(angle)
            sin = torch.sin(angle)

            if axis == 0:  # X-axis rotation
                rot_matrix = torch.tensor([[1, 0, 0],
                                           [0, cos, -sin],
                                           [0, sin, cos]], device=self.device)
            elif axis == 1:  # Y-axis rotation
                rot_matrix = torch.tensor([[cos, 0, sin],
                                           [0, 1, 0],
                                           [-sin, 0, cos]], device=self.device)
            else:  # Z-axis rotation
                rot_matrix = torch.tensor([[cos, -sin, 0],
                                           [sin, cos, 0],
                                           [0, 0, 1]], device=self.device)

            # Apply affine grid transform
            grid = F.affine_grid(rot_matrix.unsqueeze(0), volume.shape, align_corners=True)
            volume = F.grid_sample(volume, grid, align_corners=True, mode='bilinear')

        return volume

    def translate3d(self, volume: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """
        Translate 3D volume using PyTorch

        Parameters:
            volume: 4D tensor (1, 1, D, H, W)
            shifts: Translation values for each axis (3,)
        """
        # Create translation matrix
        trans_matrix = torch.eye(4, device=self.device)
        trans_matrix[:3, 3] = shifts

        # Create affine grid
        grid = F.affine_grid(trans_matrix[:3].unsqueeze(0), volume.shape, align_corners=True)

        # Apply translation
        return F.grid_sample(volume, grid, align_corners=True, mode='bilinear')

    def add_rician_noise(self, volume: torch.Tensor, std: float) -> torch.Tensor:
        """
        Add Rician noise to the volume using PyTorch

        Parameters:
            volume: Input tensor
            std: Standard deviation of the noise
        """
        # Generate complex noise
        noise_real = torch.randn_like(volume, device=self.device) * std
        noise_imag = torch.randn_like(volume, device=self.device) * std

        # Add noise and compute magnitude
        noisy_real = volume + noise_real
        noisy_imag = noise_imag

        return torch.sqrt(noisy_real ** 2 + noisy_imag ** 2)

    def adjust_intensity(self, volume: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply gamma correction to volume"""
        return volume.pow(gamma)

    def basic_augment(self, volume: torch.Tensor, params: Dict) -> Tuple[torch.Tensor, Dict]:
        """Apply basic geometric augmentations"""
        augmented = volume.clone()

        if params.get('rotate'):
            angles = torch.tensor(
                random.uniform(-10, 10),
                device=self.device
            )
            augmented = self.rotate3d(augmented, angles)
            params['rotation_angles'] = angles.cpu().tolist()

        if params.get('flip'):
            if random.random() > 0.5:
                augmented = torch.flip(augmented, dims=[2])  # Flip along X-axis
                params['flipped_axis'] = 0

        if params.get('translate'):
            shifts = torch.tensor(
                random.randint(-10, 10),
                device=self.device
            )
            augmented = self.translate3d(augmented, shifts)
            params['shifts'] = shifts.cpu().tolist()

        return augmented, params

    def advanced_augment(self, volume: torch.Tensor, params: Dict) -> Tuple[torch.Tensor, Dict]:
        """Apply noise and intensity augmentations (source images only)"""
        augmented = volume.clone()

        if params.get('noise'):
            std = random.uniform(0.01, 0.03)
            augmented = self.add_rician_noise(augmented, std)
            params['noise_std'] = float(std)

        if params.get('intensity'):
            gamma = random.uniform(0.8, 1.2)
            augmented = self.adjust_intensity(augmented, gamma)
            params['gamma'] = float(gamma)

        return augmented, params

    def generate_augmentations(self, source_image: np.ndarray,
                               target_image: np.ndarray,
                               base_filename: str,
                               num_augmentations: int = 40) -> None:
        """
        Generate augmented pairs using GPU acceleration
        """
        # Convert numpy arrays to PyTorch tensors
        source_tensor = torch.from_numpy(source_image).float().to(self.device)
        target_tensor = torch.from_numpy(target_image).float().to(self.device)

        # Add batch and channel dimensions
        source_tensor = source_tensor.unsqueeze(0).unsqueeze(0)
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)

        for i in range(num_augmentations):
            # Initialize parameters
            geometric_params = {
                'rotate': True,
                'flip': True,
                'translate': True,
                'augmentation_id': i
            }

            source_specific_params = {
                'noise': True,
                'intensity': True
            }

            # Apply geometric augmentations
            aug_source, params_source = self.basic_augment(
                source_tensor, geometric_params.copy()
            )
            aug_target, params_target = self.basic_augment(
                target_tensor, geometric_params.copy()
            )

            # Apply source-specific augmentations
            aug_source, params_source = self.advanced_augment(
                aug_source, params_source
            )

            # Convert back to numpy arrays
            aug_source_np = aug_source.squeeze().cpu().numpy()
            aug_target_np = aug_target.squeeze().cpu().numpy()

            # Save augmented images
            aug_filename = f"{base_filename}_aug_{i:03d}.nii.gz"

            nib.save(
                nib.Nifti1Image(aug_source_np, np.eye(4)),
                self.source_dir / aug_filename
            )
            nib.save(
                nib.Nifti1Image(aug_target_np, np.eye(4)),
                self.target_dir / aug_filename
            )

            # Store metadata
            self.augmentation_metadata[aug_filename] = params_source

        # Save metadata
        with open(self.save_dir / 'augmentation_metadata.json', 'w') as f:
            json.dump(self.augmentation_metadata, f, indent=4)


class BatchGPUMRIAugmenter:
    """Handles batch processing of MRI images using GPU acceleration"""

    def __init__(self, source_dir: Path, target_dir: Path,
                 save_dir: Path, device: Optional[str] = None):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.augmenter = GPUMRIContrastiveAugmenter(save_dir, device)
        self.tester = MRIAugmentationTester(
            self.augmenter.source_dir,
            self.augmenter.target_dir
        )

    def process_directories(self, num_augmentations: int = 40,
                            batch_size: int = 4) -> None:
        """
        Process all images in batches using GPU acceleration
        """
        source_files = list(self.source_dir.glob('*.nii.gz'))
        target_files = list(self.target_dir.glob('*.nii.gz'))

        logger.info(f"Found {len(source_files)} source and "
                    f"{len(target_files)} target images")

        # Process in batches
        for i in range(0, len(source_files), batch_size):
            batch_source_files = source_files[i:i + batch_size]
            batch_target_files = random.sample(target_files, len(batch_source_files))

            for source_file, target_file in tqdm(
                    zip(batch_source_files, batch_target_files),
                    desc=f"Processing batch {i // batch_size + 1}"
            ):
                try:
                    source_img = nib.load(source_file).get_fdata()
                    target_img = nib.load(target_file).get_fdata()

                    self.augmenter.generate_augmentations(
                        source_img,
                        target_img,
                        source_file.stem,
                        num_augmentations
                    )

                except Exception as e:
                    logger.error(f"Error processing {source_file}: {str(e)}")
                    continue

        # Run tests after processing
        self.run_tests()


def main():
    # Example usage
    source_dir = Path("./T1-organized/source_1.5T")
    target_dir = Path("./T1-organized/target_3T")
    save_dir = Path("./data/augmented")

    # Create batch augmenter with GPU support
    batch_augmenter = BatchGPUMRIAugmenter(
        source_dir,
        target_dir,
        save_dir,
        device='cuda'  # Explicitly use GPU
    )

    # Process all images
    batch_augmenter.process_directories(num_augmentations=40, batch_size=4)


if __name__ == "__main__":
    main()
