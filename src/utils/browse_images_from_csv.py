import pandas as pd
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
import warnings

def browse_images(csv_path, images_folder):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    images_folder = Path(images_folder)
    if not images_folder.exists():
        print(f"Error: Images folder not found at {images_folder}")
        return

    # Build list of image paths and validate
    image_ids = df["image_id"].astype(str).tolist()
    image_paths = []
    invalid_count = 0
    
    for img_id in image_ids:
        img_path = images_folder / f"{img_id}"
        print(img_path)
        if img_path.exists():
            image_paths.append(img_path)
        else:
            invalid_count += 1
    if invalid_count > 0:
        warnings.warn(f"Warning: {invalid_count} image path(s) not found and will be skipped")
    
    if not image_paths:
        print("Error: No valid images found in the specified folder")
        return

    # GUI setup
    root = tk.Tk()
    root.title("Image Browser")

    index = {"value": 0}

    img_label = tk.Label(root)
    img_label.pack()

    id_label = tk.Label(root, font=("Arial", 16))
    id_label.pack()

    def show_image():
        try:
            img_path = image_paths[index["value"]]

            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")

            img = Image.open(img_path)
            img = img.resize((500, 500))
            tk_img = ImageTk.PhotoImage(img)

            img_label.config(image=tk_img)
            img_label.image = tk_img

            id_label.config(text=f"Image ID: {img_path.stem}")
        except FileNotFoundError as e:
            id_label.config(text=f"Error: {str(e)}")
            img_label.config(image='')
            img_label.image = None
        except Exception as e:
            id_label.config(text=f"Error loading image: {str(e)}")
            img_label.config(image='')
            img_label.image = None

    def next_image():
        try:
            if index["value"] < len(image_paths) - 1:
                index["value"] += 1
                show_image()
        except Exception as e:
            print(f"Error navigating to next image: {e}")

    def prev_image():
        try:
            if index["value"] > 0:
                index["value"] -= 1
                show_image()
        except Exception as e:
            print(f"Error navigating to previous image: {e}")

    tk.Button(root, text="← Previous", command=prev_image).pack(side="left", padx=20, pady=10)
    tk.Button(root, text="Next →", command=next_image).pack(side="right", padx=20, pady=10)

    root.bind("<Left>", lambda e: prev_image())
    root.bind("<Right>", lambda e: next_image())

    try:
        show_image()
        root.mainloop()
    except Exception as e:
        print(f"Error starting image browser: {e}")
        root.destroy()


csv_path = r"C:\MyData\Projects\Computer vision\project version 1\data\raw\1.csv"
images_folder = r"C:\MyData\Projects\Computer vision\project version 1\data\processed\lbp"
    
try:
    browse_images(csv_path, images_folder)
except Exception as e:
    print(f"Fatal error: {e}")
