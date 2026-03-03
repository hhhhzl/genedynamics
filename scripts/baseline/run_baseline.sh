#!/bin/bash
# Phase 0: Run baseline experiments. Results go to config output_dir (e.g. results/single2d/mdcoas-f).
# After running, use freeze_baseline.sh to copy results to a timestamped baseline dir for regression.
#
# Usage:
#   ./scripts/baseline/run_baseline.sh           # run all
#   ./scripts/baseline/run_baseline.sh smoke     # smoke: minimal set (1 config per env)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

run_one() {
  echo "[baseline] Running $1"
  python -m genedynamics.experiments.runner "$1" || true
}

if [ "${1:-}" = "smoke" ]; then
  echo "[baseline] Smoke run (minimal set)"
  run_one configs/single_2d/mdcoas-f.yaml
  run_one configs/d3il_avoiding/dpcc_9d.yaml
  run_one configs/d3il_avoiding/safediffuser_9d.yaml
  exit 0
fi

echo "[baseline] Full run"

# single_2d
for cfg in configs/single_2d/mbd.yaml configs/single_2d/ebmbd.yaml configs/single_2d/mdoc.yaml \
           configs/single_2d/mdcoas.yaml configs/single_2d/mdcoas-a.yaml configs/single_2d/mdcoas-f.yaml; do
  run_one "$cfg"
done

# d3il_avoiding
for cfg in configs/d3il_avoiding/ebmbd.yaml configs/d3il_avoiding/mdoc.yaml configs/d3il_avoiding/mdcoas.yaml \
           configs/d3il_avoiding/mdcoas-f.yaml configs/d3il_avoiding/dpcc.yaml configs/d3il_avoiding/dpcc_9d.yaml \
           configs/d3il_avoiding/safediffuser.yaml configs/d3il_avoiding/safediffuser_9d.yaml; do
  run_one "$cfg"
done

echo "[baseline] Done. Run ./scripts/baseline/freeze_baseline.sh to save baseline for regression."
