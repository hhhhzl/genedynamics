# Isaac Lab deploy image for genedynamics.
#
# Based on NVIDIA's official Isaac Lab image; installs genedynamics on top.
# Isaac Lab is too heavy (~20 GB) for the default dev image, so it lives in
# its own Dockerfile stage. Build with:
#
#   docker build -f docker/isaac-lab.Dockerfile -t genedynamics-isaac-lab .
#
# Run tests:
#
#   docker run --gpus all genedynamics-isaac-lab \
#       python -m pytest tests/deploy/test_isaac_lab_io.py -m isaac_lab
#
# Phase 18 — see deploy/PHASE_16_18_PLAN.md for context.

# --------------------------------------------------------------------------
# Base: NVIDIA Isaac Lab image (includes Isaac Sim + torch + CUDA runtime)
# --------------------------------------------------------------------------
ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:4.5.0
FROM ${ISAAC_LAB_IMAGE} AS base

# --------------------------------------------------------------------------
# Install genedynamics
# --------------------------------------------------------------------------
WORKDIR /workspace/genedynamics

# Copy only dependency files first for layer caching.
COPY pyproject.toml setup.cfg setup.py* ./
COPY genedynamics/__init__.py genedynamics/__init__.py

RUN pip install --no-cache-dir -e ".[deploy]" 2>/dev/null || \
    pip install --no-cache-dir -e . || true

# Copy the full source tree.
COPY . .

# Re-install with full source so entry points and editable mode work.
RUN pip install --no-cache-dir -e ".[deploy]" 2>/dev/null || \
    pip install --no-cache-dir -e .

# --------------------------------------------------------------------------
# Default entrypoint: run Isaac Lab deploy tests
# --------------------------------------------------------------------------
CMD ["python", "-m", "pytest", "tests/deploy/test_isaac_lab_io.py", "-m", "isaac_lab", "-v"]
