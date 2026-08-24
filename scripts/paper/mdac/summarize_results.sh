#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python -m genedynamics.experiments.utils.metrics summarize \
  results/arm/surface_scan results/arm/peg_insert results/humanoid/push_to_line \
  --output reports/mdac
