import os
from PIL import Image, ImageFilter
import cv2
import numpy as np
import pywt

# -------------------------------------------------
# Helper conversions
# -------------------------------------------------

def pil_to_gray_np(img):
    return np.array(img.convert("L"))

def np_to_pil(arr):
    return Image.fromarray(arr)

# -------------------------------------------------
# FILTER DEFINITIONS
# -------------------------------------------------

def clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    gray = pil_to_gray_np(img)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe_obj.apply(gray)
    return np_to_pil(result)

def filter_edge_enhance(img):
    return img.filter(ImageFilter.EDGE_ENHANCE)

def gaussian_blur(img, ksize=5):
    gray = pil_to_gray_np(img)
    result = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return np_to_pil(result)

def median_filter(img, ksize=5):
    gray = pil_to_gray_np(img)
    result = cv2.medianBlur(gray, ksize)
    return np_to_pil(result)

def canny(img, low=50, high=150):
    gray = pil_to_gray_np(img)
    edges = cv2.Canny(gray, low, high)
    return np_to_pil(edges)

def sobel(img):
    gray = pil_to_gray_np(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return np_to_pil(cv2.convertScaleAbs(mag))

def lbp(img):
    gray = pil_to_gray_np(img)
    h, w = gray.shape
    lbp_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):
            center = gray[i, j]
            code = 0
            code |= (gray[i-1, j-1] > center) << 7
            code |= (gray[i-1, j]   > center) << 6
            code |= (gray[i-1, j+1] > center) << 5
            code |= (gray[i, j+1]   > center) << 4
            code |= (gray[i+1, j+1] > center) << 3
            code |= (gray[i+1, j]   > center) << 2
            code |= (gray[i+1, j-1] > center) << 1
            code |= (gray[i, j-1]   > center)
            lbp_img[i, j] = code

    return np_to_pil(lbp_img)

def gabor(img, ksize=21, sigma=5, theta=0, lambd=10, gamma=0.5):
    gray = pil_to_gray_np(img)
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
    result = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    return np_to_pil(result)

def wavelet_transform(img, wavelet='db2'):
    gray = pil_to_gray_np(img)
    LL, (LH, HL, HH) = pywt.dwt2(gray, wavelet)

    # Normalize each band to 0–255
    def norm(a):
        a = a - a.min()
        a = a / (a.max() + 1e-6)
        return (a * 255).astype(np.uint8)

    return {
        "LL": np_to_pil(norm(LL)),
        "LH": np_to_pil(norm(LH)),
        "HL": np_to_pil(norm(HL)),
        "HH": np_to_pil(norm(HH)),
    }

def polar_transform(img):
    gray = pil_to_gray_np(img)
    center = (gray.shape[1]//2, gray.shape[0]//2)
    max_radius = min(center)
    result = cv2.warpPolar(gray, (360, max_radius), center, max_radius, cv2.WARP_FILL_OUTLIERS)
    return np_to_pil(result)

def rotational_difference(img):
    gray = pil_to_gray_np(img)
    rotated = cv2.rotate(gray, cv2.ROTATE_180)
    diff = cv2.absdiff(gray, rotated)
    return np_to_pil(diff)

def bilateral(img, d=9, sigma_color=75, sigma_space=75):
    gray = pil_to_gray_np(img)
    result = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    return np_to_pil(result)

def laplacian_of_gaussian(img, ksize=5):
    gray = pil_to_gray_np(img)
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    log = cv2.Laplacian(blur, cv2.CV_64F)
    return np_to_pil(cv2.convertScaleAbs(log))

def adaptive_threshold(img):
    gray = pil_to_gray_np(img)
    result = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return np_to_pil(result)

# -------------------------------------------------
# FILTER REGISTRY
# -------------------------------------------------

FILTERS = {
    "edge_enhance": filter_edge_enhance,
    "clahe": clahe,
    "gaussian_blur": gaussian_blur,
    "median_filter": median_filter,
    "canny": canny,
    "sobel": sobel,
    "lbp": lbp,
    "gabor": gabor,
    "wavelet_transform": wavelet_transform,
    "polar_transform": polar_transform,
    "rotational_difference": rotational_difference,
    "bilateral": bilateral,
    "laplacian_of_gaussian": laplacian_of_gaussian,
    "adaptive_threshold": adaptive_threshold
}

# -------------------------------------------------
# PROCESSING PIPELINE
# -------------------------------------------------

def process_images(input_folder, output_folder, limit=None):
    # Create output folders
    for filter_name in FILTERS.keys():
        os.makedirs(os.path.join(output_folder, filter_name), exist_ok=True)

    images = [f for f in os.listdir(input_folder)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if limit:
        images = images[:limit]

    print(f"Processing {len(images)} images...")

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path)

        for filter_name, filter_func in FILTERS.items():
            result = filter_func(img)

            base, ext = os.path.splitext(img_name)

            # Wavelet returns 4 images
            if filter_name == "wavelet_transform":
                for band, band_img in result.items():
                    out_name = f"{base}_{filter_name}_{band}{ext}"
                    band_img.save(os.path.join(output_folder, filter_name, out_name))
            else:
                out_name = f"{base}_{filter_name}{ext}"
                result.save(os.path.join(output_folder, filter_name, out_name))

    print("Done!")

# -----------------------------
# RUN
# -----------------------------

INPUT_FOLDER = r"C:\MyData\Projects\Computer vision\project version 1\data\raw\train_images"
OUTPUT_FOLDER = r"C:\MyData\Projects\Computer vision\project version 1\data\processed"

LIMIT = 200   # <--- change this to control how many images to process

process_images(INPUT_FOLDER, OUTPUT_FOLDER, LIMIT)
