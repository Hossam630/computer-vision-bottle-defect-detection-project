import cv2
import os

# Input and output folder paths
input_folder = "C:\\MyData\\Projects\\Computer vision\\project version 1\\data\\raw\\train_images"
output_folder = "C:\\MyData\\Projects\\Computer vision\\project version 1\\data\\processed\\canny_filter"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Set how many images you want to process
image_limit = 1000   # change this number as needed

count = 0

for filename in os.listdir(input_folder):
    if count >= image_limit:
        print("Image limit reached.")
        break

    if filename.lower().endswith(extensions):
        img_path = os.path.join(input_folder, filename)

        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        # Apply Canny edge detection
        edges = cv2.Canny(img, 100, 200)

        # Save result using the SAME filename
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, edges)

        count += 1
        print(f"Processed ({count}/{image_limit}): {filename}")

print("Done.")
