#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python_bin="${MGA_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  if command -v python >/dev/null 2>&1; then python_bin="python"; else python_bin="python3"; fi
fi
"$python_bin" -m genedynamics.experiments.utils.metrics summarize \
  results/arm/surface_scan results/arm/peg_insert results/humanoid/push_to_line \
  --output reports/mga
