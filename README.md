# Bikefitting
MVP project

## Requirements

- **Python 3.8 or newer**
- **CUDA 12.4** (for GPU acceleration)

## Installation

### Option 1: Quick Install (Recommended)

Use the provided installation script which handles all dependencies including PyTorch CUDA wheels:

```bash
# Create and activate a conda environment (recommended)
conda create -n bikefitting python=3.12
conda activate bikefitting

# Run the installation script
./install.sh
```

### Option 2: Manual Installation

1. Create and activate a virtual environment:

   **Windows:**
   ```bash
   python -m venv bikeEnv
   bikeEnv\Scripts\activate
   ```

   **Mac/Linux:**
   ```bash
   python -m venv bikeEnv
   source bikeEnv/bin/activate
   ```

2. Install PyTorch with CUDA support:
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
   ```

3. Install remaining requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the sam3 package:
   ```bash
   pip install -e ./sam3
   ```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```