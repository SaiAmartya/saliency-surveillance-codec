#!/usr/bin/env bash
# One-shot setup for the project on macOS.
# Creates a Python venv, installs deps, and verifies ffmpeg is reachable.
#
# Usage:  chmod +x setup.sh && ./setup.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "==> Project root: $PROJECT_DIR"

# 1) ffmpeg (libx264 + libx265) via Homebrew
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> ffmpeg not found; installing via Homebrew..."
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install it first:  https://brew.sh/"; exit 1
  fi
  brew install ffmpeg
else
  echo "==> ffmpeg already installed: $(ffmpeg -version | head -1)"
fi

# 2) Python venv
if [ ! -d ".venv" ]; then
  echo "==> Creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) Install Python deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4) Quick import smoke test
python -c "import cv2, torch, numpy, skimage; print('numpy', numpy.__version__, '| cv2', cv2.__version__, '| torch', torch.__version__)"

echo
echo "==> Setup complete. Activate with:  source .venv/bin/activate"
echo "==> Then:  python scripts/make_test_video.py && python scripts/ablation_bgfg.py --quick --saliency yolo+spectral"
