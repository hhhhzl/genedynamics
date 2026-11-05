#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.runtime/baseline_tasks_torch/venv/bin/python}"
export CROSS_EBDM_NATIVE_LIBRARY="${CROSS_EBDM_NATIVE_LIBRARY:-$ROOT/.runtime/cross_ebdm/native/libcross_ebdm.so}"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "four-task transfer confirmation requires physical GPU1 via CUDA_VISIBLE_DEVICES=1" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=1
exec "$PYTHON" scripts/paper/cross_ebdm/run_confirmation.py "$@"
