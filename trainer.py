import os
import toml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pathlib import Path
from datetime import datetime

from data import MRIDataModule
from model import CycleGan


class MRITrainer:
    """
    Trainer class that handles the complete training pipeline for MRI image translation.
    Reads configuration from TOML file and sets up training environment.
    """

    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration from TOML file.

        Args:
            config_path: Path to the TOML configuration file
        """
        # Load configuration
        self.config = toml.load(config_path)

        # Create necessary directories
        self._create_directories()

        # Set up PyTorch Lightning training components
        self.datamodule = self._create_datamodule()
        self.model = self._create_model()
        self.callbacks = self._create_callbacks()
        self.logger = self._create_logger()
        self.trainer = self._create_trainer()

    def _create_directories(self):
        """Create necessary directories for logs and checkpoints"""
        # Create paths with timestamps to prevent overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(self.config['paths']['log_dir']) / timestamp
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir']) / timestamp

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save configuration file for reproducibility
        config_save_path = self.log_dir / 'config.toml'
        with open(config_save_path, 'w') as f:
            toml.dump(self.config, f)

    def _create_datamodule(self) -> MRIDataModule:
        """Initialize the MRI DataModule with configuration parameters"""
        return MRIDataModule(
            source_dir=self.config['paths']['source_dir'],
            target_dir=self.config['paths']['target_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            aug_prob=self.config['augmentation']['probability']
        )

    def _create_model(self) -> CycleGan:
        """Initialize the CycleGAN model with configuration parameters"""
        return CycleGan(
            generator_config=self.config['generator'],
            discriminator_config=self.config['discriminator']
        )

    def _create_callbacks(self) -> list:
        """Create training callbacks for checkpointing and early stopping"""
        callbacks = []

        # Checkpoint callback to save best models
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename='mri_cyclegan_{epoch:03d}_{val_generator_total_loss:.3f}',
            monitor='val_generator_total_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)

        # Early stopping callback to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_generator_total_loss',
            patience=10,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping)

        return callbacks

    def _create_logger(self) -> TensorBoardLogger:
        """Create TensorBoard logger"""
        return TensorBoardLogger(
            save_dir=self.log_dir,
            name='mri_translation',
            version='cyclegan'
        )

    def _create_trainer(self) -> pl.Trainer:
        """Create PyTorch Lightning trainer with configuration parameters"""
        return pl.Trainer(
            accelerator=self.config['training']['accelerator'],
            devices=self.config['training']['devices'],
            max_epochs=self.config['training']['max_epochs'],
            callbacks=self.callbacks,
            logger=self.logger,
            accumulate_grad_batches=self.config['training']['accumulate_grad_batches'],
            precision=self.config['training']['precision'],
            log_every_n_steps=10
        )

    def train(self):
        """Start the training process"""
        print("Starting training...")
        self.trainer.fit(self.model, self.datamodule)
        print(f"Training completed. Logs saved to: {self.log_dir}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize and start training
    trainer = MRITrainer("./options/config.toml")
    trainer.train()