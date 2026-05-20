#!/usr/bin/env bash
# Build pydiffvg from source (pip install git+... does NOT work for diffvg).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFVG_DIR="${DIFFVG_DIR:-$SCRIPT_DIR/third_party/diffvg}"

detect_cuda_home() {
  if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "$CUDA_HOME"
    return 0
  fi
  local candidate
  for candidate in \
    /usr/local/cuda-12.6 \
    /usr/local/cuda-12.4 \
    /usr/local/cuda-12.1 \
    /usr/local/cuda-12 \
    /usr/local/cuda; do
    if [ -x "$candidate/bin/nvcc" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

setup_conda_toolchain() {
  if ! command -v conda >/dev/null 2>&1; then
    return 0
  fi

  echo "==> Ensuring conda C++ runtime is new enough (GLIBCXX_3.4.30+)..."
  conda install -y -c conda-forge libstdcxx-ng gcc_linux-64 gxx_linux-64 cmake >/dev/null

  if [ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++" ]; then
    export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
    echo "==> Using conda compilers:"
    echo "    CC=$CC"
    echo "    CXX=$CXX"
  fi
}

echo "==> Checking PyTorch + CUDA before building pydiffvg..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available. Fix torch first, then rebuild pydiffvg.\n"
        "Expected: torch 2.4.0+cu124 and cuda: True"
    )
print("cuda: True")
PY

setup_conda_toolchain

if ! CUDA_HOME="$(detect_cuda_home)"; then
  echo "ERROR: Could not find nvcc. Set CUDA_HOME manually, e.g.:"
  echo "  export CUDA_HOME=/usr/local/cuda-12.4"
  exit 1
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "==> Using CUDA toolkit: $CUDA_HOME"
nvcc --version | head -n 4
echo "==> Host compiler: $(${CXX:-g++} --version | head -n 1)"

if [ ! -d "$DIFFVG_DIR/.git" ]; then
  echo "==> Cloning diffvg into $DIFFVG_DIR ..."
  mkdir -p "$(dirname "$DIFFVG_DIR")"
  git clone https://github.com/BachiLi/diffvg.git "$DIFFVG_DIR"
fi

cd "$DIFFVG_DIR"
echo "==> Updating submodules..."
git submodule update --init --recursive

echo "==> Installing diffvg Python deps..."
pip install svgwrite svgpathtools cssutils torch-tools

echo "==> Cleaning previous diffvg build cache..."
rm -rf build/ dist/ *.egg-info

echo "==> Building and installing pydiffvg (this may take a few minutes)..."
pip uninstall -y diffvg 2>/dev/null || true
python setup.py install

echo "==> Verifying pydiffvg (from Pipeline dir, not diffvg source dir)..."
cd "$SCRIPT_DIR"
python - <<'PY'
import torch
import pydiffvg
pydiffvg.set_use_gpu(torch.cuda.is_available())
print("pydiffvg OK")
PY

echo "Done."
