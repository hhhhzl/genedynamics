#!/usr/bin/env bash
# Setup genedynamics environment. Optional third-party: d3il, softzoo, 3dgs, all
# Usage: ./scripts/setup/setup.sh [d3il|softzoo|3dgs|all]
# Default (no args): d3il + pip install -e . + PyTorch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

_target="${1:-d3il}"

case "$_target" in
  d3il)
    echo "=== Setup D3IL ==="
    "$SCRIPT_DIR/setup_d3il.sh"
    pip install -e .
    pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall 2>/dev/null || pip install torch
    python -c "import jax; print(jax.devices())" 2>/dev/null || true
    python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || true
    ;;
  softzoo)
    echo "=== Setup SoftZoo assets ==="
    "$SCRIPT_DIR/setup_softzoo.sh"
    ;;
  3dgs)
    echo "=== Setup 3DGS ==="
    echo "Set GAUSSIAN_SPLATTING_PATH to official 3DGS repo. See scripts/tasks/3dgs/README.md"
    ;;
  all)
    "$SCRIPT_DIR/setup.sh" d3il
    "$SCRIPT_DIR/setup.sh" softzoo
    ;;
  *)
    echo "Usage: $0 [d3il|softzoo|3dgs|all]"
    echo "  d3il    - D3IL + genedynamics + PyTorch (default)"
    echo "  softzoo - SoftZoo assets (data/softzoo/assets)"
    echo "  3dgs    - 3DGS (set GAUSSIAN_SPLATTING_PATH)"
    echo "  all     - d3il + softzoo"
    exit 1
    ;;
esac
