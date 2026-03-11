#!/bin/bash
# Run MRMFMBD (Soft-robot S1+S3) experiments
# Output: results/mrmfmbd/softzoo_flat/

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${VENV:-$ROOT/.venv_arm64/bin/python}"

echo "=== MRMFMBD (Soft-robot S1+S3) ==="
echo ""

echo ">>> SoftZoo flat (caterpillar, levels 0)"
echo "    Use stub when SoftZoo not installed (use_stub_when_unavailable: true)"
$PY -m genedynamics.experiments.runner configs/mrmfmbd/softzoo_flat.yaml
echo ""

echo "=== Output ==="
echo "Results: results/mrmfmbd/softzoo_flat/"
find results/mrmfmbd -type f 2>/dev/null || true
echo ""
echo "Done."
