import os
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
from src.preprocessing.transforms import get_transforms
from src.models.model_factory import build_model

def load_model(model_name, num_classes, weights_path, device):
    model = build_model(model_name, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Custom dataset for flat test image directory (no class subfolders)
class FlatImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        self.image_paths.sort()  # Optional: sort for reproducibility

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

def get_dataloader(data_dir, batch_size, num_workers, phase="val", flat_test=True):
    transforms = get_transforms()
    if flat_test:
        dataset = FlatImageFolderDataset(data_dir, transform=transforms.get(phase, transforms["val"]))
    else:
        dataset = datasets.ImageFolder(data_dir, transform=transforms.get(phase, transforms["val"]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader, dataset

def apply_model(model, dataloader, device, class_names=None, flat_test=True):
    predictions = []
    image_paths = []
    with torch.no_grad():
        for batch in dataloader:
            if flat_test:
                inputs, paths = batch
            else:
                inputs, _ = batch
                paths = None
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            if flat_test:
                image_paths.extend(paths)
    if flat_test:
        # Just return image paths and predicted indices
        return list(zip(image_paths, predictions))
    else:
        # Map predictions to class names
        image_paths = [path for path, _ in dataloader.dataset.samples]
        pred_class_names = [class_names[p] for p in predictions]
        return list(zip(image_paths, pred_class_names))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply trained model to dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory (flat folder for test images)")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth model weights file")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (as used in build_model)")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV file for predictions")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model_name, args.num_classes, args.weights, device)
    # Use flat_test=True for test images in a flat directory
    dataloader, dataset = get_dataloader(args.data_dir, args.batch_size, args.num_workers, flat_test=True)
    # class_names is not used for flat test, just output predicted indices
    results = apply_model(model, dataloader, device, class_names=None, flat_test=True)

    # Save predictions to CSV
    import csv
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_class_index"])
        for img_path, pred_class in results:
            writer.writerow([img_path, pred_class])
    print(f"Predictions saved to {args.output_csv}")


# running script:
# python src/apply.py --data_dir path/to/data --weights path/to/model.pth --model_name MODEL_NAME --num_classes N --output_csv predictions.csv

#python -m src.apply --data_dir "/home/hossam-bassyoni/Computer vision/project_v1/data/raw/test_images" --weights "/home/hossam-bassyoni/Computer vision/project_v1/outputs/checkpoints/best_resnet50.pth" --model_name resnet50 --num_classes 2 --output_csv predictions.csv

