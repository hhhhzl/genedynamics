#!/bin/bash
# Quick run: gsplat warm start + MBD with Phase 3+4 (prior_bbox, best_chain, train/test metrics)
# Run from repo root.
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

echo "Step 1: Train gsplat for warm start (6k iters, ~30s)..."
python scripts/tasks/3dgs/train_gsplat.py configs/3dgs/main/lego_mbd_canonical.yaml \
  --output results/3dgs/lego_gsplat_warmstart --iters 6000 --n-gaussians 128

echo "Step 2: Run MBD with warm start + Phase 3+4 (~2min)..."
python scripts/tasks/3dgs/run_full_experiment.py configs/3dgs/main/lego_mbd_quality_recovery_vramfit.yaml \
  --initial-scene-path results/3dgs/lego_gsplat_warmstart/scene_params.npz \
  --initialization-mode prior_center --init-jitter-scale 0.1 \
  --n-seeds 2 --best-chain

echo ""
echo "Metrics:"
cat results/3dgs/main/lego_mbd_quality_recovery_vramfit/metrics.json 2>/dev/null || true
echo ""
echo "Expected: test_psnr ~10.3 dB, train_psnr ~10.2 dB"
