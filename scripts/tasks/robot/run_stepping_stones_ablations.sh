#!/bin/bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

echo "=== Stepping Stones 2D: 2GO ablations ==="
for cfg in \
  configs/quadruped/stepping_stones_2d/ablations/2go_no_gate.yaml \
  configs/quadruped/stepping_stones_2d/ablations/2go_no_tail.yaml \
  configs/quadruped/stepping_stones_2d/ablations/2go_no_probe.yaml \
  configs/quadruped/stepping_stones_2d/ablations/2go_no_retract.yaml
do
  echo ">>> $cfg"
  $PY -m genedynamics.experiments.runner "$cfg"
  echo ""
done

echo "Done."

