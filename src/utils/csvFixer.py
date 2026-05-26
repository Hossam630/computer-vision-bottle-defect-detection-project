import pandas as pd

# Load the CSV
df = pd.read_csv("/home/hossam-bassyoni/Computer vision/project_v1/predictions.csv")

# Keep only the last part of the path
df["image_path"] = df["image_path"].apply(lambda p: p.split("/")[-1])

# Save the updated CSV
df.to_csv("output.csv", index=False)
