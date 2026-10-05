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
  configs/arm/peg_insert/main/mga.yaml \
  configs/arm/peg_insert/ablation/no_rl_prior.yaml \
  configs/arm/peg_insert/ablation/no_retraction.yaml \
  configs/arm/peg_insert/baseline/issa.yaml \
  configs/arm/peg_insert/baseline/atacom.yaml \
  configs/arm/peg_insert/baseline/mppi.yaml \
  configs/arm/peg_insert/baseline/dial.yaml \
  configs/arm/peg_insert/baseline/pegasusflow.yaml \
  configs/humanoid/push_to_line/main \
  configs/humanoid/push_to_line/baseline \
  --require-visuals
"$python_bin" -m genedynamics.experiments.utils.metrics verify \
  configs/arm/surface_scan/ablation \
  --require-visuals
"$python_bin" -m genedynamics.experiments.utils.metrics verify \
  configs/humanoid/push_to_line/ablation/no_rl_prior.yaml \
  configs/humanoid/push_to_line/ablation/no_learned_reliability.yaml \
  configs/humanoid/push_to_line/ablation/no_controllability_geometry.yaml \
  configs/humanoid/push_to_line/ablation/no_retraction.yaml \
  --require-visuals
"$python_bin" -m genedynamics.experiments.utils.metrics verify \
  configs/humanoid/push_to_line/ablation/no_stiffness.yaml \
  --suites force_regulation_15n force_regulation_30n --require-visuals
