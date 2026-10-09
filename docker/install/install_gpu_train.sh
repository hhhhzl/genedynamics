#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/workspace}"

python -m pip install --upgrade pip "setuptools>=77" wheel
python -m pip install -e "${PROJECT_ROOT}[simulation,optimization,reports,test]" \
  "jax[cuda12]==0.6.2" "brax==0.14.1" "mujoco==3.6.0" "mujoco-mjx==3.6.0" \
  "urchin>=0.0.27"
# Torch supports optional policy/training paths; the shared planner remains JAX.
python -m pip install "torch==2.10.0" "torchvision==0.25.0" \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip check
