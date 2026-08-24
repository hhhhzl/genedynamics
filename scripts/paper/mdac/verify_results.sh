#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python -m genedynamics.experiments.utils.metrics audit \
  configs/arm/surface_scan configs/arm/peg_insert configs/humanoid/push_to_line
python -m genedynamics.experiments.utils.metrics verify \
  configs/arm/surface_scan/main configs/arm/surface_scan/baseline \
  configs/arm/peg_insert configs/humanoid/push_to_line \
  --require-visuals
python -m genedynamics.experiments.utils.metrics verify \
  configs/arm/surface_scan/ablation \
  --suites hybrid_stripes hybrid_center_hard hybrid_center_soft \
  --require-visuals
