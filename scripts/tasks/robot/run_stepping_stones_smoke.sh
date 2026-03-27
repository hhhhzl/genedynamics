#!/bin/bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

echo "=== Stepping Stones 2D: smoke ==="
for cfg in \
  configs/quadruped/stepping_stones_2d/smoke/mbd_smoke.yaml \
  configs/quadruped/stepping_stones_2d/smoke/mdoc_smoke.yaml \
  configs/quadruped/stepping_stones_2d/smoke/mdcoas_smoke.yaml
do
  echo ">>> $cfg"
  $PY -m genedynamics.experiments.runner "$cfg"
  echo ""
done

echo "Done."

