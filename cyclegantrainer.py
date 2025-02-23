# train.py
import os
import toml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from model import CycleMRIGAN
from data import MRIDataModule2D


def train_model(config):
    # Set random seeds for reproducibility
    pl.seed_everything(config['training']['seed'])

    # Initialize data module
    data_module = MRIDataModule2D(
        train_source_dir=config['data']['train_source_dir'],
        train_target_dir=config['data']['train_target_dir'],
        valid_source_dir=config['data']['valid_source_dir'],
        valid_target_dir=config['data']['valid_target_dir'],
        batch_size=config['data']['batch_size'],
        stats_file=config['data']['stats_file'],
        num_workers=config['data']['num_workers'],
        source_rotation_angle=config['augmentation']['source_rotation_angle'],
        source_translation=config['augmentation']['source_translation'],
        target_rotation_angle=config['augmentation']['target_rotation_angle'],
        target_translation=config['augmentation']['target_translation']
    )

    # Initialize model
    config['model']['max_epochs'] = config['training']['max_epochs']
    model = CycleMRIGAN(config['model'])

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(config['training']['checkpoint_dir'], 'checkpoints'),
            filename='{epoch}-{val_generator_total_loss:.2f}',
            monitor='val_generator_total_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config['training']['log_dir'],
        name=config['training']['experiment_name']
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['training']['log_every_n_steps']
    )

    # Train model
    trainer.fit(model, data_module)

if __name__ == '__main__':
    # Load configuration file
    config = toml.load('./options/config_2d.toml')
    train_model(config)