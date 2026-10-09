#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/workspace}"

# Keep the image on the V1 numerical stack. Install dependencies through the
# package extras so the image cannot silently drift beyond the JAX support range.
python -m pip install --upgrade pip "setuptools>=77,<81" wheel
python -m pip install -e "${PROJECT_ROOT}[simulation,optimization,reports,test]" \
  "jax[cpu]==0.6.2" "brax==0.14.1" "mujoco==3.6.0" "mujoco-mjx==3.6.0" \
  "urchin>=0.0.27"
python -m pip install "torch>=2.3,<3" --index-url https://download.pytorch.org/whl/cpu
# Optional parity oracle for DIAL; runtime uses the bundled spline cache.
python -m pip install "jax-cosmo==0.1.0"
python -m pip check
