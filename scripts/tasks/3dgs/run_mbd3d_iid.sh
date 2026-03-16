#!/bin/bash
# Run Ours-MBD (iid) on NeRF Synthetic via experiment runner

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

OBJECT="${1:-lego}"
CONFIG="configs/3dgs/${OBJECT}_mbd_iid.yaml"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "=== Ours-MBD (iid) on $OBJECT ==="
echo "Config: $CONFIG"
echo ""

$PY -m genedynamics.experiments.runner "$CONFIG"

echo ""
echo "Results: results/3dgs/${OBJECT}_mbd_iid/"
