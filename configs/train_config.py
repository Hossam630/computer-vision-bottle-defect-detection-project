# configs/train_config.py

CONFIG = {
    "data_dir": "/home/hossam-bassyoni/Computer vision/project_v1/data/raw/data_dir",  # All training data in class subfolders here
    "batch_size": 32,
    "num_workers": 4,
    "num_classes": 2,
    "num_epochs": 40,
    "learning_rate": 1e-4,
    "model_name": "vgg16",  # resnet50, vgg16, vit_b_16
    "checkpoint_dir": "outputs/checkpoints",
    "device": "cpu", # "cuda" if torch.cuda.is_available() else "cpu"
    "n_splits": 5,  # Number of folds for k-fold cross-validation
}
