#!/bin/bash
# Installation script for BikeFitting project
# This script installs all dependencies with compatible versions
# this script was made with Claude Optus 4.5 assistance 

set -e  # Exit on error

echo "=========================================="
echo "BikeFitting Installation Script"
echo "=========================================="

# Check if we're in a conda environment or virtual environment
if [[ -z "$CONDA_DEFAULT_ENV" && -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: No conda or virtual environment detected."
    echo "It's recommended to install in a virtual environment."
    read -p "Continue anyway? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

echo ""
echo "Step 1: Installing PyTorch with CUDA 12.4 support..."
echo "----------------------------------------------"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "Step 2: Installing xformers..."
echo "----------------------------------------------"
pip install xformers==0.0.29.post1

echo ""
echo "Step 3: Installing other dependencies..."
echo "----------------------------------------------"
pip install numpy==1.26.0
pip install opencv-python==4.9.0.80
pip install opencv-python-headless==4.9.0.80
pip install pandas
pip install ultralytics
pip install huggingface_hub

echo ""
echo "Step 4: Installing sam3 package..."
echo "----------------------------------------------"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/sam3" ]]; then
    pip install -e "$SCRIPT_DIR/sam3"
else
    echo "WARNING: sam3 directory not found. Skipping sam3 installation."
fi

echo ""
echo "Step 5: Verifying installation..."
echo "----------------------------------------------"
pip check && echo "✓ All dependencies are compatible!" || echo "⚠ Some dependency conflicts detected."

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Installed versions:"
python -c "
import torch
import torchvision
import numpy
import cv2
print(f'  torch:       {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  numpy:       {numpy.__version__}')
print(f'  opencv:      {cv2.__version__}')
try:
    import xformers
    print(f'  xformers:    {xformers.__version__}')
except:
    print('  xformers:    (not available)')
"

echo ""
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
