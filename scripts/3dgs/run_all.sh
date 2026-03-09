#!/bin/bash
# Run all 3DGS experiments: Ours-MBD (iid/corr) on lego + chair

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

echo "=== 3DGS Experiment 1: Small object 3D reconstruction ==="
echo ""

for obj in lego chair; do
  echo ">>> Ours-MBD (iid) on $obj"
  $PY -m genedynamics.experiments.runner "configs/3dgs/${obj}_mbd_iid.yaml" || true
  echo ""
  echo ">>> Ours-MBD (corr) on $obj"
  $PY -m genedynamics.experiments.runner "configs/3dgs/${obj}_mbd_corr.yaml" || true
  echo ""
done

echo "=== Output ==="
echo "results/3dgs/lego_mbd_iid/"
echo "results/3dgs/lego_mbd_corr/"
echo "results/3dgs/chair_mbd_iid/"
echo "results/3dgs/chair_mbd_corr/"
echo ""
echo "3DGS-MAP baseline (manual): ./scripts/3dgs/run_3dgs_map.sh data/nerf_synthetic/lego results/3dgs/lego_3dgs_map"
echo ""
echo "Done."
