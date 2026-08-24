#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
while IFS= read -r config; do
  if [[ "$config" == */ablation/* ]]; then
    python -m genedynamics.experiments.runner "$config" \
      --suites hybrid_stripes hybrid_center_hard hybrid_center_soft
  else
    python -m genedynamics.experiments.runner "$config"
  fi
done < <(find configs/arm/surface_scan -type f -name '*.yaml' ! -name '_base.yaml' | sort)
