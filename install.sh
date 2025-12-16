#!/usr/bin/env bash
set -euo pipefail

# Run this after you have activated your conda/venv.
# By default it installs CUDA 12.4 PyTorch wheels.
# Set BIKEFITTING_CPU_ONLY=1 to force CPU wheels.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install --upgrade pip

if [[ "${BIKEFITTING_CPU_ONLY:-0}" == "1" ]]; then
  python -m pip install "torch==2.5.1" "torchvision==0.20.1"
else
  python -m pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu124
fi

python -m pip install -r "$SCRIPT_DIR/requirements.txt"

OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
if [[ "$OS_NAME" == "Linux" ]]; then
  if [[ -d "$SCRIPT_DIR/sam3" ]]; then
    python -m pip install -e "$SCRIPT_DIR/sam3"
  fi
else
  echo "Skipping sam3 install (sam3 requires Linux). Detected OS: $OS_NAME"
fi

python -m pip check || true
