#!/bin/bash
# Phase 0: Freeze current results/ as baseline for regression.
# Usage:
#   ./scripts/baseline/freeze_baseline.sh [OUTPUT_DIR]
#   Default: results_baseline_<YYYYMMDD_HHMM>
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DEST="${1:-results_baseline_$(date +%Y%m%d_%H%M)}"
SRC="results"

if [ ! -d "$SRC" ]; then
  echo "[freeze] No $SRC/ found. Run experiments first."
  exit 1
fi

echo "[freeze] Copying $SRC/ -> $DEST/"
cp -r "$SRC" "$DEST"
echo "[freeze] Baseline frozen at $DEST"
echo "[freeze] To compare later: python scripts/baseline/compare_regression.py $DEST results"
