#!/usr/bin/env bash
# Environment setup: AORender + Intrinsic + pydiffvg
# Tested: NVIDIA driver 575.x (CUDA 12.9), Python 3.10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Installing PyTorch 2.4 + cu124 (compatible with driver CUDA 12.9)..."
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing AORender Python dependencies..."
pip install -r requirements-base.txt

echo "==> Building pydiffvg against current PyTorch..."
bash install_diffvg.sh

echo "==> Installing Intrinsic (torch version locked via constraints.txt)..."
pip install -c constraints.txt git+https://github.com/compphoto/Intrinsic.git

echo "==> Verifying installation..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
import pydiffvg
import diffusers
import segment_anything
import intrinsic
print("pydiffvg, diffusers, sam, intrinsic: OK")
PY

echo ""
echo "Done. Run pipeline with:"
echo "  CUDA_VISIBLE_DEVICES=0 python pipeline.py --run_all --generate_albedo --image_name xxx.png"
