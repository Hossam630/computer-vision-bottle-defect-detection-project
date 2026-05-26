import cv2
import os
import numpy as np

# --- CONFIGURE YOUR FOLDERS ---
input_folder = "C:\\MyData\\Projects\\Computer vision\\project version 1\\data\\processed\\canny_filter"
output_folder = "C:\\MyData\\Projects\\Computer vision\\project version 1\\data\\processed\\hough_circle_transform"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # --- HOUGH CIRCLE TRANSFORM ---
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=200
        )

        # Draw detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        # Save processed image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

        print(f"Processed: {filename}")

print("Done.")
