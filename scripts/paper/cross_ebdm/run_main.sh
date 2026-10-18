#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.runtime/cross_ebdm/venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi

configs=(
  configs/arm/obstacle_transfer/main/cross_ebdm.yaml
  configs/arm/guided_contact_alignment/main/cross_ebdm.yaml
  configs/arm/keyed_insertion_recovery/main/cross_ebdm.yaml
  configs/quadruped/mobile_keyed_insertion_recovery/main/cross_ebdm.yaml
)

for config in "${configs[@]}"; do
  "$PYTHON" -m genedynamics.experiments.runner "$config" "$@"
done
