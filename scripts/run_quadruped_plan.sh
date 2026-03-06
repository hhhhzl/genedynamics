#!/bin/bash
# Run quadruped pure plan: 2 methods (MBD, MD-COAS) x 3 levels
# Output: results/quadruped/flat/mbd_plan, results/quadruped/flat/mdcoas_plan
# Trajectory: results/quadruped/plan_*/level_*/seed_*/trajectory_3d.png

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

echo "=== Quadruped pure plan: MBD + cfsmbd_full x 3 levels ==="
echo ""

echo ">>> MBD (levels 0,1,2)"
$PY -m genedynamics.experiments.runner configs/quadruped/flat/mbd_plan.yaml
echo ""

echo ">>> MD-COAS (levels 0,1,2)"
$PY -m genedynamics.experiments.runner configs/quadruped/flat/mdcoas_plan.yaml
echo ""

echo "=== Trajectory outputs ==="
echo "MBD:    results/quadruped/flat/mbd_plan/level_*/seed_0/trajectory_3d.png"
echo "MD-COAS: results/quadruped/flat/mdcoas_plan/level_*/seed_0/trajectory_3d.png"
find results/quadruped/flat -name "trajectory_3d.png" 2>/dev/null || true
echo ""
echo "Done."
