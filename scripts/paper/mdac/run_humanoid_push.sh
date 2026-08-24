#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
while IFS= read -r config; do
  python -m genedynamics.experiments.runner "$config"
done < <(find configs/humanoid/push_to_line -type f -name '*.yaml' ! -name '_base.yaml' | sort)
