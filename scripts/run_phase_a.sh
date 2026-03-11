#!/bin/bash
# Phase A: 系统打通
# crawling only, single mode, single fidelity
# Goal: evaluator + solver + logging 都通

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# VENV=path or .venv_phase_a (created by Phase A setup) or conda fedguide
PY="${VENV:-$ROOT/.venv_phase_a/bin/python}"
if ! [ -x "$PY" ] && command -v conda &>/dev/null; then
  PY="conda run -n fedguide python"
fi

echo "=== Phase A: 系统打通 ==="
echo "  config: configs/co_design/phase_a_smoke.yaml"
echo "  task: crawling_ground, 1 mode, 1 fidelity"
echo ""

$PY scripts/run_co_design.py configs/co_design/phase_a_smoke.yaml --seed 0

echo ""
echo "=== Output ==="
echo "  results/phase_a/smoke/results.json"
echo "  results/phase_a/smoke/experiment.jsonl"
echo ""
echo "Done."
