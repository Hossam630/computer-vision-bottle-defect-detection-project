from sklearn.model_selection import train_test_split
# New function: split dataset into train/val using train_test_split
def create_split_dataloaders(data_dir, batch_size, num_workers, val_ratio=0.2, random_state=42):
    """
    Splits the dataset in data_dir into training and validation sets using train_test_split,
    without requiring separate folders. Returns dataloaders and dataset sizes dicts.
    """
    transforms = get_transforms()
    dataset = datasets.ImageFolder(data_dir, transform=transforms["train"])
    targets = [s[1] for s in dataset.samples]
    train_idx, val_idx = train_test_split(
        list(range(len(targets))),
        test_size=val_ratio,
        stratify=targets,
        random_state=random_state
    )
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    # Set transforms for each subset
    train_subset.dataset.transform = transforms["train"]
    val_subset.dataset.transform = transforms["val"]
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_idx), "val": len(val_idx)}
    print("Class mapping:", dataset.class_to_idx)
    return dataloaders, dataset_sizes
# src/dataloader/dataloaders.py

import os
from torch.utils.data import DataLoader
from torchvision import datasets
from src.preprocessing.transforms import get_transforms


from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold

def create_dataloaders(data_dir, batch_size, num_workers):
    # Legacy function for two folders: train/val
    transforms = get_transforms()
    datasets_dict = {
        phase: datasets.ImageFolder(
            os.path.join(data_dir, phase),
            transform=transforms[phase]
        )
        for phase in ["train", "val"]
    }
    dataloaders = {
        phase: DataLoader(
            datasets_dict[phase],
            batch_size=batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=num_workers
        )
        for phase in ["train", "val"]
    }
    dataset_sizes = {phase: len(datasets_dict[phase]) for phase in ["train", "val"]}
    print("Class mapping:", datasets_dict["train"].class_to_idx)
    return dataloaders, dataset_sizes

def create_kfold_dataloaders(data_dir, batch_size, num_workers, n_splits=5, random_state=42):
    """
    Returns a generator of (train_loader, val_loader, train_size, val_size) for each fold.
    Assumes all data is in data_dir, organized by class subfolders.
    """
    transforms = get_transforms()
    dataset = datasets.ImageFolder(data_dir, transform=transforms["train"])
    targets = [s[1] for s in dataset.samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print("Class mapping:", dataset.class_to_idx)
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.samples, targets)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        # For val, use val transforms
        train_subset.dataset.transform = transforms["train"]
        val_subset.dataset.transform = transforms["val"]
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        yield {
            "fold": fold,
            "dataloaders": {"train": train_loader, "val": val_loader},
            "dataset_sizes": {"train": len(train_idx), "val": len(val_idx)}
        }
