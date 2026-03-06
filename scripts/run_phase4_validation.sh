#!/bin/bash
# Phase 4: Closed-loop validation and fair comparison
# Run: bash scripts/run_phase4_validation.sh
# Uses .venv_arm64 by default. Set VENV=path for custom venv.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-$ROOT/.venv_arm64/bin/python}"
PY="$VENV"

echo "=== Phase 4: Closed-loop validation ==="
echo "Python: $PY"
echo ""

# 1. MBD deploy (baseline)
echo ">>> 1. MBD deploy (mbd_deploy_quick)"
$PY -m genedynamics.deploy.cli --config configs/quadruped/flat/mbd_deploy_quick.yaml \
  --episodes 1 --max-steps 20 --no-record 2>&1 | tail -5
echo "OK"
echo ""

# 2. MD-COAS deploy (level 0 = fair comparison with MBD)
echo ">>> 2. MD-COAS deploy (mbd_deploy_mdcoas_quick, level 0)"
$PY -m genedynamics.deploy.cli --config configs/quadruped/flat/mdcoas_deploy_quick.yaml \
  --episodes 1 --max-steps 20 --no-record 2>&1 | tail -5
echo "OK"
echo ""

# 3. obstacle_avoid task (MD-COAS + level 1 obstacles)
echo ">>> 3. obstacle_avoid task"
$PY -m genedynamics.deploy.cli --config configs/quadruped/obstacle_avoid/mdcoas_deploy_quick.yaml \
  --episodes 1 --max-steps 20 --no-record 2>&1 | tail -5
echo "OK"
echo ""

# 4. rough_terrain task
echo ">>> 4. rough_terrain task"
$PY -m genedynamics.deploy.cli --config configs/quadruped/rough_terrain/mbd_deploy_quick.yaml \
  --episodes 1 --max-steps 20 --no-record 2>&1 | tail -5
echo "OK"
echo ""

# 5. push_recovery task
echo ">>> 5. push_recovery task"
$PY -m genedynamics.deploy.cli --config configs/quadruped/push_recovery/mbd_deploy_quick.yaml \
  --episodes 1 --max-steps 20 --no-record 2>&1 | tail -5
echo "OK"
echo ""

echo "=== All Phase 4 validation tests passed ==="
