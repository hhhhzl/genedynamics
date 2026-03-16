#!/bin/bash
# Paper-like MBD3D run: larger bridge budget + JAX on-demand GPU memory.
# Run from repo root: bash scripts/tasks/3dgs/run_mbd3d_paperlike.sh
set -e

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

WARM_OUT="results/3dgs/lego_gsplat_48g_stable"

echo "Step 1/3: gsplat warm start (fixed Gaussians)"
python -u scripts/tasks/3dgs/train_gsplat.py configs/3dgs/main/lego_mbd_warmup.yaml \
  --output "$WARM_OUT" --iters 3000 --n-gaussians 4096 --mask-loss

echo "Step 2/3: gsplat refine to >25 dB (target)"
python -u scripts/tasks/3dgs/train_gsplat.py configs/3dgs/main/lego_mbd_warmup.yaml \
  --output "$WARM_OUT" --iters 1500 \
  --from-checkpoint "$WARM_OUT/scene_params.npz" --refine-lr 0.001 --mask-loss

echo "Step 3/3: MBD bridge with paper-like K/M and JAX on-demand memory"
python -u scripts/tasks/3dgs/run_full_experiment.py configs/3dgs/main/lego_mbd_trainingfree.yaml \
  --initial-scene-path "$WARM_OUT/scene_params.npz" \
  --max-initial-gaussians 1024 \
  --initialization-mode direct --init-jitter-scale 0.0 --n-seeds 1 \
  --jax-preallocate false --jax-allocator platform --cuda-malloc-async

echo ""
echo "Warmstart metrics:"
python - <<'PY'
import json, pathlib
p = pathlib.Path("results/3dgs/lego_gsplat_48g_stable/metrics.json")
print(json.dumps(json.loads(p.read_text()), indent=2) if p.exists() else "missing")
PY

echo ""
echo "MBD metrics:"
python - <<'PY'
import json, pathlib
p = pathlib.Path("results/3dgs/main/lego_mbd_trainingfree/metrics.json")
print(json.dumps(json.loads(p.read_text()), indent=2) if p.exists() else "missing")
PY
