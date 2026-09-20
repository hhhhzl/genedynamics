#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
full_suite_configs=(
  configs/arm/surface_scan/main/mga.yaml
  configs/arm/surface_scan/baseline/model_based_only.yaml
  configs/arm/surface_scan/baseline/standalone_rl.yaml
  configs/arm/surface_scan/baseline/dial.yaml
  configs/arm/surface_scan/baseline/mppi.yaml
  configs/arm/surface_scan/baseline/pegasusflow.yaml
  configs/arm/surface_scan/baseline/issa.yaml
  configs/arm/surface_scan/baseline/atacom.yaml
  configs/arm/surface_scan/ablation/no_rl_prior.yaml
  configs/arm/surface_scan/ablation/no_learned_reliability.yaml
  configs/arm/surface_scan/ablation/no_controllability_geometry.yaml
  configs/arm/surface_scan/ablation/no_retraction.yaml
)
suites=(
  rigid_plane
  rigid_cylinder
  rigid_convex
  rigid_bumpy
  rigid_unseen
  soft_plane
  soft_cylinder
  soft_convex
  soft_bumpy
  soft_unseen
  hybrid_stripes
  hybrid_center_hard
  hybrid_center_soft
)
for config in "${full_suite_configs[@]}"; do
  # JAX retains one compiled executable family per surface geometry. Running
  # all thirteen suites in one interpreter eventually exhausts the CPU Docker
  # memory limit, so each suite gets a fresh process. The final resume-only
  # pass restores the complete protocol manifest and aggregate summary.
  for suite in "${suites[@]}"; do
    python -m genedynamics.experiments.runner "$config" --resume \
      --suite "$suite" "$@"
  done
  python -m genedynamics.experiments.runner "$config" --resume "$@"
done
