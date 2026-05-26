# src/train.py


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
