#!/usr/bin/env bash

set -euo pipefail

ROOT="${1:-/workspace}"

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  "numpy>=1.26.0" \
  "scipy>=1.13.0" \
  "pyyaml>=6.0.1" \
  "tqdm>=4.65.0" \
  "trimesh==4.10.1" \
  "pybullet==3.2.7" \
  "imageio>=2.37.2" \
  "einops>=0.8.2" \
  "jinja2>=3.1.0" \
  "matplotlib>=3.8.0" \
  "seaborn>=0.13.0" \
  "pandas>=2.2.0" \
  "gym>=0.26.2" \
  "gymnasium>=0.29.1" \
  "mujoco>=3.1.3" \
  "mujoco-mjx" \
  "dm-control>=1.0.14" \
  "gin-config>=0.5.0" \
  "pybind11>=3.0.0" \
  "pytest>=9.0.2" \
  "black>=24.3.0" \
  "flake8>=7.0.0"

python -m pip install \
  "torch==2.10.0" \
  "torchvision==0.25.0" \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install -U "jax[cuda12]" "brax==0.14.1"

# Editable install without pulling the repo's full mixed dependency set twice.
python -m pip install --no-deps -e "${ROOT}"
