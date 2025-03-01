import os
import argparse
import toml
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from model import ContrastiveTraining
from data import MRIDataModule2D


def main(config):
    # Set seed for reproducibility
    pl.seed_everything(config["training"]["seed"])

    # Create experiment directory
    experiment_dir = Path(config["paths"]["output_dir"]) / config["experiment"]["name"]
    experiment_dir.mkdir(exist_ok=True, parents=True)

    # Save config to experiment directory
    with open(experiment_dir / "config.toml", "w") as f:
        toml.dump(config, f)

    # Initialize data module
    data_module = MRIDataModule2D(
        train_source_dir=config["data"]["train_source_dir"],
        train_target_dir=config["data"]["train_target_dir"],
        valid_source_dir=config["data"]["valid_source_dir"],
        valid_target_dir=config["data"]["valid_target_dir"],
        stats_file=config["data"]["stats_file"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        source_rotation_angle=config["augmentation"]["source_rotation_angle"],
        source_translation=config["augmentation"]["source_translation"],
        target_rotation_angle=config["augmentation"]["target_rotation_angle"],
        target_translation=config["augmentation"]["target_translation"]
    )

    # Initialize model
    # Update configuration with save directory
    generator_config = config["generator"]
    generator_config["save_dir"] = str(experiment_dir / "visualizations")

    model = ContrastiveTraining(
        generator_config=generator_config,
        discriminator_config=config["discriminator"]
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=config["training"]["monitor_metric"],
        dirpath=experiment_dir / "checkpoints",
        filename="{epoch:02d}-{val_total_loss:.4f}",
        save_top_k=config["training"]["save_top_k"],
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor=config["training"]["monitor_metric"],
        patience=config["training"]["patience"],
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=str(experiment_dir),
        name="logs"
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config["training"]["devices"],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        precision=config["training"]["precision"],
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        log_every_n_steps=config["training"].get("log_every_n_steps", 50)
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save the final model
    trainer.save_checkpoint(experiment_dir / "checkpoints" / "final_model.ckpt")

    print(f"Training completed! Model saved to {experiment_dir / 'checkpoints'}")


if __name__ == "__main__":
    config = toml.load(str(Path("./options/config_cut.toml").absolute()))
    main(config)