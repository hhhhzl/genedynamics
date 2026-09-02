#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
configs=(
  configs/arm/peg_insert/main/mga.yaml
  configs/arm/peg_insert/ablation/no_rl_prior.yaml
  configs/arm/peg_insert/ablation/no_learned_reliability.yaml
  configs/arm/peg_insert/baseline/model_based_only.yaml
  configs/arm/peg_insert/baseline/standalone_rl.yaml
  configs/arm/peg_insert/baseline/dial.yaml
  configs/arm/peg_insert/baseline/mppi.yaml
  configs/arm/peg_insert/baseline/pegasusflow.yaml
  configs/arm/peg_insert/baseline/issa.yaml
  configs/arm/peg_insert/baseline/atacom.yaml
)
for config in "${configs[@]}"; do
  python -m genedynamics.experiments.runner "$config" --resume
done
