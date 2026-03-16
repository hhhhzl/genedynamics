#!/bin/bash
# Run quadruped pure plan: MBD, MD-COAS, MBD_go2 x 3 levels
# Output: results/quadruped/flat/{mbd_plan,mdcoas_plan,mbd_plan_go2}

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

echo "=== Quadruped pure plan: MBD + MD-COAS + MBD_go2 x 3 levels ==="
echo ""

echo ">>> MBD (Ant, levels 0,1,2)"
$PY -m genedynamics.experiments.runner configs/quadruped/flat/mbd_plan.yaml
echo ""

echo ">>> MD-COAS (Ant, levels 0,1,2)"
$PY -m genedynamics.experiments.runner configs/quadruped/flat/mdcoas_plan.yaml
echo ""

echo ">>> MBD Go2 (Unitree Go2, levels 0,1,2) - requires mujoco-menagerie"
$PY -m genedynamics.experiments.runner configs/quadruped/flat/mbd_plan_go2.yaml 2>/dev/null || echo "  (skip if MUJOCO_MENAGERIE_PATH not set)"
echo ""

echo "=== Trajectory outputs ==="
echo "MBD:     results/quadruped/flat/mbd_plan/level_*/seed_0/trajectory_3d.png"
echo "MD-COAS: results/quadruped/flat/mdcoas_plan/level_*/seed_0/trajectory_3d.png"
echo "MBD_go2: results/quadruped/flat/mbd_plan_go2/level_*/seed_0/trajectory_3d.png"
find results/quadruped/flat -name "trajectory_3d.png" 2>/dev/null || true
echo ""
echo "Done."
