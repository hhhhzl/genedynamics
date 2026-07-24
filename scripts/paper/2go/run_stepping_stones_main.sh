#!/bin/bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

echo "=== Stepping Stones 2D: main experiments ==="
for cfg in \
  configs/quadruped/stepping_stones_2d/main/mbd.yaml \
  configs/quadruped/stepping_stones_2d/main/mppi.yaml \
  configs/quadruped/stepping_stones_2d/main/ebmbd.yaml \
  configs/quadruped/stepping_stones_2d/main/mdoc.yaml \
  configs/quadruped/stepping_stones_2d/main/mdcoas.yaml \
  configs/quadruped/stepping_stones_2d/main/twogo.yaml
do
  echo ">>> $cfg"
  python -m genedynamics.experiments.runner "$cfg"
  echo ""
done

echo "Done."
