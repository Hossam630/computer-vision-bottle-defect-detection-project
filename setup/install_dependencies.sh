#!/usr/bin/env bash

echo "=============================================="
echo " Installing Bottle Inspection Project Packages"
echo "=============================================="

# ---- Python version check ----
PYTHON_VERSION=$(python3 - <<EOF
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)

echo "Detected Python version: $PYTHON_VERSION"

# ---- Upgrade pip ----
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# ---- Core ML dependencies ----
echo "Installing core ML packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ---- Training utilities ----
echo "Installing training utilities..."
pip install numpy matplotlib tqdm scikit-learn

# ---- Vision Transformer dependencies ----
echo "Installing ViT dependencies..."
pip install timm

# ---- Logging & experiment tracking ----
echo "Installing logging tools..."
pip install tensorboard

# ---- YAML config support ----
echo "Installing config utilities..."
pip install pyyaml

# ---- Optional: image handling ----
echo "Installing image processing tools..."
pip install pillow opencv-python

# ---- Optional: Jupyter for experimentation ----
echo "Installing Jupyter..."
pip install jupyterlab

echo "=============================================="
echo " All dependencies installed successfully!"
echo "=============================================="
