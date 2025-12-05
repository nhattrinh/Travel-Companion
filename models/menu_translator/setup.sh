#!/bin/bash
# Setup script for Menu Translator ML models

set -e

echo "=== Menu Translator Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version - modify for GPU)
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8), use instead:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For GPU (CUDA 12.1), use instead:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install timm
echo "Installing timm..."
pip install timm

# Install HuggingFace Transformers for translation
echo "Installing HuggingFace Transformers..."
pip install transformers sentencepiece sacremoses

# Install PaddleOCR
echo "Installing PaddleOCR..."
pip install paddlepaddle paddleocr

# Install other dependencies
echo "Installing other dependencies..."
pip install pillow numpy opencv-python langdetect prometheus-client tqdm

# Create data directories
echo "Creating data directories..."
mkdir -p data/train
mkdir -p data/val
mkdir -p data/test
mkdir -p data/ocr_eval
mkdir -p checkpoints

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the demo (mock translation):"
echo "     python main.py --mode demo"
echo ""
echo "  3. Run with REAL translation (downloads ~600MB NLLB model):"
echo "     python main.py --mode demo --real-translation"
echo ""
echo "  4. For training food classifier, add images to:"
echo "     data/train/<class>/ and data/val/<class>/"
echo "     Then run: python main.py --mode train --epochs 10"
echo ""
echo "  5. For evaluation:"
echo "     python main.py --mode eval"
