#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

python -m genedynamics.deploy.cli \
  --config configs/quadruped/stepping_stones_2d_exec/deploy/walk_follow_sim.yaml
