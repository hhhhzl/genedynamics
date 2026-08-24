#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$script_dir/run_surface_scan.sh"
"$script_dir/run_peg_insert.sh"
"$script_dir/run_humanoid_push.sh"
"$script_dir/render_results.sh"
"$script_dir/summarize_results.sh"
"$script_dir/verify_results.sh"
