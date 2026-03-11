#!/bin/bash
# Phase 4: Fair comparison - MBD vs MD-COAS (same env, level 0 obstacles)
# Both use quadruped_flat_mjx, Nsample=128, Ndiffuse=30, 20 steps.
# Run: bash scripts/tasks/soft_robot/run_phase4_fair_comparison.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

echo "=== Fair comparison: MBD vs MD-COAS (level 0) ==="
echo ""

echo ">>> MBD"
$PY -m genedynamics.deploy.cli --config configs/quadruped/flat/mbd_deploy_quick.yaml \
  --episodes 1 --max-steps 20 --no-record 2>&1 | grep -E "Episode|Done|Error"
echo ""

echo ">>> MD-COAS (level 0 obstacles = same as MBD)"
$PY -m genedynamics.deploy.cli --config configs/quadruped/flat/mdcoas_deploy_quick.yaml \
  --episodes 1 --max-steps 20 --no-record 2>&1 | grep -E "Episode|Done|Error"
echo ""

echo "=== Fair comparison complete ==="
