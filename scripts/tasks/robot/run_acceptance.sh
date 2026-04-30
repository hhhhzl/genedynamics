#!/bin/bash
# Run acceptance tests (closed-loop validation)
# Uses .venv_arm64 by default. Set VENV=path for custom Python.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

echo "=== 1. Quadruped MBD deploy ==="
$PY -m genedynamics.deploy.cli --config configs/quadruped/flat/mbd_deploy_quick.yaml \
  --episodes 1 --max-steps 15 --no-record
echo "OK"

echo "=== 2. Quadruped MD-COAS deploy ==="
$PY -m genedynamics.deploy.cli --config configs/quadruped/flat/mdcoas_deploy_quick.yaml \
  --episodes 1 --max-steps 15 --no-record
echo "OK"

echo "=== 3. Quadruped experiment (mbd.yaml) ==="
$PY -m genedynamics.experiments.runner configs/quadruped/flat/mbd_plan.yaml --level 0 --seed 0
echo "OK"

echo "=== 4. UAV experiment (mbd.yaml) ==="
$PY -m genedynamics.experiments.runner configs/uav3d/mbd.yaml --level 0 --seed 0
echo "OK"

echo "=== All acceptance tests passed ==="
