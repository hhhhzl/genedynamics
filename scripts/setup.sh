#!/usr/bin/env bash
set -euo pipefail

# Keep this file as the stable entrypoint for local setup helpers.
# (Some users have this hard-coded in their workflow.)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[setup] If you want to use D3IL, run:"
echo "  ${HERE}/setup_d3il.sh"


