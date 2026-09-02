#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
configs=(
  configs/arm/surface_scan/main/mga.yaml
  configs/arm/surface_scan/baseline/model_based_only.yaml
  configs/arm/surface_scan/baseline/standalone_rl.yaml
  configs/arm/surface_scan/baseline/dial.yaml
  configs/arm/surface_scan/baseline/mppi.yaml
  configs/arm/surface_scan/baseline/pegasusflow.yaml
  configs/arm/surface_scan/baseline/issa.yaml
  configs/arm/surface_scan/baseline/atacom.yaml
  configs/arm/surface_scan/ablation/no_controllability_geometry.yaml
  configs/arm/surface_scan/ablation/no_retraction.yaml
  configs/arm/surface_scan/ablation/no_stiffness.yaml
  configs/arm/surface_scan/ablation/fixed_or_euclidean_stiffness.yaml
)
for config in "${configs[@]}"; do
  if [[ "$config" == */ablation/* ]]; then
    python -m genedynamics.experiments.runner "$config" --resume \
      --suites hybrid_stripes hybrid_center_hard hybrid_center_soft
  else
    python -m genedynamics.experiments.runner "$config" --resume
  fi
done
