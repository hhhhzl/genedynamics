#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/run_main.sh" "$@"
"$HERE/run_baselines.sh" "$@"
"$HERE/run_ablation.sh" "$@"
