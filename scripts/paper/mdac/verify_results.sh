#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python -m genedynamics.experiments.utils.metrics audit \
  configs/arm/surface_scan configs/arm/peg_insert configs/humanoid/push_to_line
python -m genedynamics.experiments.utils.metrics verify \
  configs/arm/surface_scan configs/arm/peg_insert configs/humanoid/push_to_line \
  --require-visuals
