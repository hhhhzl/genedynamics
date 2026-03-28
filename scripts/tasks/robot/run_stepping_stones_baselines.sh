#!/bin/bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

echo "=== Stepping Stones 2D baselines ==="
echo ">>> MBD"
$PY -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/main/mbd.yaml
echo ""
echo ">>> MDOC"
$PY -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/main/mdoc.yaml
echo ""
echo ">>> MD-COAS (CFSMBD Full)"
$PY -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/main/mdcoas.yaml
echo ""
echo "Done."

