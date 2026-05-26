# computer-vision-bottle-defect-detection-project

This project trains and applies deep learning models (ResNet50, VGG16, ViT) for image classification on bottle inspection data.

## Setup

1. **Install dependencies:**
   ```sh
   bash setup/install_dependencies.sh
   ```

2. **Prepare data:**
   - Place your images and CSVs in the `data/raw/` directory.
   - Use `src/utils/images_separator.py` to organize images into class folders if needed.

## Training

Edit your configuration in [`configs/train_config.py`](configs/train_config.py).

Run training:
```sh
python src/train.py
```
- Models and logs are saved in the `outputs/` directory.

## Inference

To generate predictions on test images:
```sh
python src/apply.py \
  --data_dir data/raw/test_images \
  --weights outputs/checkpoints/best_resnet50.pth \
  --model_name resnet50 \
  --num_classes 2 \
  --output_csv predictions.csv
```

## Post-processing

To clean up the prediction CSV (keep only image file names):
```sh
python src/utils/csvFixer.py
```
- Output is saved as `output.csv`.

## Directory Structure

- `configs/` — Training configuration files
- `data/` — Raw and processed data
- `outputs/` — Model checkpoints, logs, and predictions
- `src/` — Source code (training, inference, dataloaders, etc.)
- `setup/` — Dependency installation scripts

## Requirements

- Python 3.8+
- See `setup/install_dependencies.sh` for all required packages.

## Notes

- Supported models: ResNet50, VGG16, ViT-B-16
- Training/validation split is handled automatically.
- For custom data, update paths in `configs/train_config.py`.

---

For more details, see comments in the code files.