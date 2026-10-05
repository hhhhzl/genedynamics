#!/usr/bin/env bash
# Setup the GenerativeDynamics development environment with D3IL.
# Usage: ./scripts/setup/setup.sh [d3il]
# Default (no args): d3il + pip install -e . + PyTorch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

_target="${1:-d3il}"

case "$_target" in
  d3il)
    echo "=== Setup D3IL ==="
    cd "$ROOT"
    "$SCRIPT_DIR/setup_d3il.sh"
    pip install "pybind11>=3.0.0"
    export pybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
    pip install -e .
    pip install torch==2.10.0 torchvision==0.25.0 \
      --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
    pip install -U "jax[cuda12]"
    python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA unavailable (driver/GPU not visible?)'"
    python -c "import jax; b=jax.default_backend(); assert b == 'gpu', f'JAX expected GPU backend, got {b!r}; devices={jax.devices()}'"
    ;;
  *)
    echo "Usage: $0 [d3il]"
    echo "  d3il    - D3IL + genedynamics + PyTorch (default)"
    exit 1
    ;;
esac
