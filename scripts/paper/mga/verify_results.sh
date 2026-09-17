#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python_bin="${MGA_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  if command -v python >/dev/null 2>&1; then python_bin="python"; else python_bin="python3"; fi
fi
"$python_bin" -m genedynamics.experiments.utils.metrics audit \
  configs/arm/surface_scan configs/arm/peg_insert configs/humanoid/push_to_line
"$python_bin" -m genedynamics.experiments.utils.metrics verify \
  configs/arm/surface_scan/main configs/arm/surface_scan/baseline \
  configs/arm/peg_insert configs/humanoid/push_to_line/main \
  configs/humanoid/push_to_line/baseline \
  --require-visuals
"$python_bin" -m genedynamics.experiments.utils.metrics verify \
  configs/arm/surface_scan/ablation \
  --suites hybrid_stripes hybrid_center_hard hybrid_center_soft \
  --require-visuals
"$python_bin" -m genedynamics.experiments.utils.metrics verify \
  configs/humanoid/push_to_line/ablation/no_rl_prior.yaml \
  configs/humanoid/push_to_line/ablation/no_learned_reliability.yaml \
  --suites p2_push_ood p3_unjam p4_walk_push --require-visuals
"$python_bin" -m genedynamics.experiments.utils.metrics verify \
  configs/humanoid/push_to_line/ablation/no_tangent.yaml \
  configs/humanoid/push_to_line/ablation/no_retraction.yaml \
  --suites p3_unjam --require-visuals
"$python_bin" -m genedynamics.experiments.utils.metrics verify \
  configs/humanoid/push_to_line/ablation/no_stiffness.yaml \
  --suites p1_force_15n p1_force_30n --require-visuals
