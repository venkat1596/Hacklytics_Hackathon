# train.py
import os
import toml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import CycleFreeCycleGan
from data import MRIDataModule2D


def main():
    # Load configuration
    config = toml.load('./options/config_2d.toml')

    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['visualization_dir'], exist_ok=True)

    # Initialize data module
    data_module = MRIDataModule2D(
        train_source_dir=config['paths']['train_source_dir'],
        train_target_dir=config['paths']['train_target_dir'],
        valid_source_dir=config['paths']['valid_source_dir'],
        valid_target_dir=config['paths']['valid_target_dir'],
        stats_file=config['paths']['stats_file'],
        batch_size=config['datamodule']['batch_size'],
        num_workers=config['datamodule']['num_workers'],
        source_rotation_angle=config['datamodule']['source_rotation_angle'],
        source_translation=config['datamodule']['source_translation'],
        target_rotation_angle=config['datamodule']['target_rotation_angle'],
        target_translation=config['datamodule']['target_translation']
    )

    config['generator']['max_epochs'] = config['trainer']['max_epochs']
    config['discriminator']['max_epochs'] = config['trainer']['max_epochs']
    # Initialize model
    model = CycleFreeCycleGan(
        generator_config = config['generator'],
        discriminator_config = config['discriminator'],
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['paths']['checkpoint_dir'],
        filename='cyclegan-{epoch:02d}-{val_generator_total_loss:.2f}',
        monitor='val_generator_total_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(config['paths']['checkpoint_dir'], 'logs'),
        name='mri_cyclegan'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        check_val_every_n_epoch=config['trainer']['check_val_every_n_epoch']
    )

    # Train model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()