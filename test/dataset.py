import sys
import os
# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data import MRIDataModule

# source_dir='path/to/source',
# target_dir='path/to/target',

if __name__ == "__main__":
    # Create the datamodule
    datamodule = MRIDataModule(
        source_dir='/home/venkat/Documents/Hacklytics_Hackathon/data/source-dir',
        target_dir='/home/venkat/Documents/Hacklytics_Hackathon/data/target-dir',
        batch_size=4,
        num_workers=4
    )

    # Set up the datamodule (this will create the datasets)
    datamodule.setup()

    # Get dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Example of accessing data
    for batch in train_loader:
        source_images = batch['source']  # Shape: [B, 1, H, W, D]
        target_images = batch['target']  # Shape: [B, 1, H, W, D]
        target_min = batch['target_min']
        target_max = batch['target_max']

        print(f"Source images shape: {source_images.shape}")
        print(f"Target images shape: {target_images.shape}")
        break  # Just testing one batch