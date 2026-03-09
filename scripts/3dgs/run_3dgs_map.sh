#!/bin/bash
# Run 3DGS-MAP baseline (official 3DGS training)
# Requires: GAUSSIAN_SPLATTING_PATH or gaussian-splatting repo in project root

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATASET="${1:-data/nerf_synthetic/lego}"
OUTPUT="${2:-results/3dgs/lego_3dgs_map}"
GS_PATH="${GAUSSIAN_SPLATTING_PATH:-$ROOT/gaussian-splatting}"

if [[ ! -d "$GS_PATH" ]] || [[ ! -f "$GS_PATH/train.py" ]]; then
  echo "3DGS repo not found at $GS_PATH"
  echo "Set GAUSSIAN_SPLATTING_PATH or clone: git clone https://github.com/graphdeco-inria/gaussian-splatting"
  exit 1
fi

if [[ ! -d "$DATASET" ]]; then
  echo "Dataset not found: $DATASET"
  echo "Download NeRF Synthetic from https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1"
  exit 1
fi

echo "=== 3DGS-MAP baseline ==="
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo ""

mkdir -p "$OUTPUT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

# 3DGS uses its own conda env typically; try system python if venv doesn't have deps
"$PY" -c "
from genedynamics.solvers.single.mbd3d.integrations import GaussianSplattingCLIBackend
import os
gs = GaussianSplattingCLIBackend(gs_path=os.environ.get('GAUSSIAN_SPLATTING_PATH', '$GS_PATH'))
r = gs.train('$DATASET', '$OUTPUT')
print('returncode:', r['returncode'])
if r['stderr']:
    print('stderr:', r['stderr'][-500:])
" 2>/dev/null || {
  echo "Run 3DGS training manually:"
  echo "  cd $GS_PATH"
  echo "  python train.py -s $DATASET -m $OUTPUT"
}

echo ""
echo "Done. Checkpoints: $OUTPUT"
