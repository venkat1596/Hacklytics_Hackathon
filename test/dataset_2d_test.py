import sys
import os
# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data import MRIDataModule2D



if __name__ == "__main__":
    # Create data module
    data_module = MRIDataModule2D(
        source_dir="/home/venkat/Documents/Hacklytics_Hackathon/data/ConstraintTest/trainA",
        target_dir="/home/venkat/Documents/Hacklytics_Hackathon/data/ConstraintTest/trainB",
        stats_file="./stats/dataset_stats.json",  # Specify stats file location
        batch_size=1,
        num_workers=4
    )

    # Setup the data module
    data_module.setup()

    # Get a batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    print("Source image shape:", batch['source'].shape)
    print("Target image shape:", batch['target'].shape)
    print("Source stats - min:", batch['source_global_min'], "max:", batch['source_global_max'])
    print("Target stats - min:", batch['target_global_min'], "max:", batch['target_global_max'])