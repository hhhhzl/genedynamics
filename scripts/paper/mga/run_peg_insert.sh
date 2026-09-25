#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
configs=(
  configs/arm/peg_insert/main/mga.yaml
  configs/arm/peg_insert/ablation/no_rl_prior.yaml
  configs/arm/peg_insert/ablation/no_retraction.yaml
  configs/arm/peg_insert/baseline/issa.yaml
  configs/arm/peg_insert/baseline/atacom.yaml
  configs/arm/peg_insert/baseline/mppi.yaml
  configs/arm/peg_insert/baseline/dial.yaml
  configs/arm/peg_insert/baseline/pegasusflow.yaml
)
suites=(id_wide ood_pose ood_sensing)
for config in "${configs[@]}"; do
  for suite in "${suites[@]}"; do
    python -m genedynamics.experiments.runner "$config" \
      --suite "$suite" --seeds 0 1 2 3 4 5 6 7 8 9 --resume
  done
done
