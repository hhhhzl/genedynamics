#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup_d3il] Repo root: ${ROOT_DIR}"
echo "[setup_d3il] This script does NOT install D3IL via pip."
echo "[setup_d3il] It prints the environment variables you should export so that"
echo "[setup_d3il] D3IL can be imported as environments.d3il.*"
echo

echo "Export this in your shell before running experiments that use D3IL:"
echo
echo "  export PYTHONPATH=\"${ROOT_DIR}/third_party:\${PYTHONPATH:-}\""
echo
echo "If instead you pin the *full* upstream D3IL repo at ${ROOT_DIR}/third_party/d3il,"
echo "export this as well (so python can see third_party/d3il/environments):"
echo
echo "  export PYTHONPATH=\"${ROOT_DIR}/third_party/d3il:\${PYTHONPATH:-}\""
echo

echo "Quick import check:"
echo "  python -c \"import environments.d3il; import environments.d3il.d3il_sim; print('OK')\""


