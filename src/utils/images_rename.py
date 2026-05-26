import os

folder = r"C:\MyData\Projects\Computer vision\project version 1\data\processed\wavelet_transform"

for filename in os.listdir(folder):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    name, _ = os.path.splitext(filename)
    parts = name.split("_")

    if len(parts) > 1:
        new_name = "_".join(parts[:-1]) + ".png"
        os.rename(
            os.path.join(folder, filename),
            os.path.join(folder, new_name)
        )
        print(f"Renamed: {filename} -> {new_name}")
