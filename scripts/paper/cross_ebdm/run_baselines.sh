#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.runtime/cross_ebdm/venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi

task_roots=(
  configs/arm/obstacle_transfer
  configs/arm/guided_contact_alignment
  configs/arm/keyed_insertion_recovery
  configs/quadruped/mobile_keyed_insertion_recovery
)

for task_root in "${task_roots[@]}"; do
  while IFS= read -r config; do
    "$PYTHON" -m genedynamics.experiments.runner "$config" "$@"
  done < <(find "$task_root/baseline" -maxdepth 1 -name '*.yaml' -print | sort)
done
