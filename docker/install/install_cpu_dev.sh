#!/usr/bin/env bash

set -euo pipefail

ROOT="${1:-/workspace}"

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  "numpy>=1.26.0" \
  "scipy>=1.13.0" \
  "osqp>=0.6.7.post3" \
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

# jax-cosmo is the parity/build oracle for DIAL's quadratic node spline.  Its
# 0.1.0 package still imports pkg_resources, which setuptools 81+ removed.
# Keep this compatibility pin local to the CPU development/test image; the
# cached spline keeps the runtime solver independent of jax-cosmo.
python -m pip install \
  "setuptools<81" \
  "jax[cpu]" \
  "brax==0.14.1" \
  "jax-cosmo==0.1.0"

# Editable install without pulling the repo's GPU-only dependency bundle.
python -m pip install --no-deps -e "${ROOT}"
