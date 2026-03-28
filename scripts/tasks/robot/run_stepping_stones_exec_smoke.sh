#!/usr/bin/env bash
# Path B smoke: stepping-stones plan + Go2 qpos/qvel/ctrl + batch_summary.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# Linux headless: prefer egl; fall back to osmesa when EGL/OpenGL init fails (e.g. minimal CI images).
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec python scripts/tasks/robot/run_stepping_stones_go2_exec_batch.py \
  --config configs/quadruped/stepping_stones_2d_exec/smoke/batch_smoke.yaml
