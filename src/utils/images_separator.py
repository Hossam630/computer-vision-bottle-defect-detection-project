import os
import shutil
import pandas as pd

# --- paths ---
csv_path = "/home/hossam-bassyoni/Computer vision/project_v1/data/raw/train.csv"                 # your CSV file
source_folder = "/home/hossam-bassyoni/Computer vision/project_v1/data/raw/train_images"         # folder containing all images
destination_folder = "/home/hossam-bassyoni/Computer vision/project_v1/data/raw/data_dir"   # output folder

# --- load CSV ---
df = pd.read_csv(csv_path)

# --- create destination folders ---
for t in [0, 1]:
    os.makedirs(os.path.join(destination_folder, str(t)), exist_ok=True)

# --- copy images ---
for _, row in df.iterrows():
    image_id = row["image_id"]
    target = row["target"]

    src = os.path.join(source_folder, image_id)
    dst = os.path.join(destination_folder, str(target), image_id)

    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Missing image: {src}")
