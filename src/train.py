# src/train.py


import os
import sys

# Ensure project root is on PYTHONPATH when running as: python src/train.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.train_config import CONFIG
from src.dataloader.dataloaders import create_split_dataloaders
from src.models.model_factory import build_model
from src.training.trainer import train_model

def main():
    # Use the new split dataloader function
    dataloaders, dataset_sizes = create_split_dataloaders(
        CONFIG["data_dir"],
        CONFIG["batch_size"],
        CONFIG["num_workers"],
        val_ratio=CONFIG.get("val_ratio", 0.2),
        random_state=CONFIG.get("random_state", 42)
    )
    model = build_model(CONFIG["model_name"], CONFIG["num_classes"])
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        config=CONFIG
    )
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
