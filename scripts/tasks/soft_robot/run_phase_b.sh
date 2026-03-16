#!/bin/bash
# Phase B: S1 验证
# full S1 vs no-mode, ID (crawling_ground) vs OOD (crawling_desert)
# 通过标准: OOD 更稳, responsibilities 有可解释变化

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${VENV:-$ROOT/.venv_phase_a/bin/python}"
if ! [ -x "$PY" ] && command -v conda &>/dev/null; then
  PY="conda run -n fedguide python"
fi

echo "=== Phase B: S1 验证 ==="
echo ""

for cfg in configs/co_design/phase_b_s1_id.yaml configs/co_design/phase_b_s1_ood.yaml \
           configs/co_design/phase_b_nomode_id.yaml configs/co_design/phase_b_nomode_ood.yaml; do
  echo ">>> $cfg"
  $PY scripts/tasks/soft_robot/run_co_design.py "$cfg"
  echo ""
done

echo "=== Output ==="
echo "  results/phase_b/s1_id/results.json"
echo "  results/phase_b/s1_ood/results.json"
echo "  results/phase_b/nomode_id/results.json"
echo "  results/phase_b/nomode_ood/results.json"
echo ""
echo "Done."
