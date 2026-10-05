# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04


# Keep the interpreter line fixed while following the supported official slim
# variant.  This also lets a clean arm64 Docker Desktop rebuild the CPU image.
FROM python:${PYTHON_VERSION}-slim AS base-cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa \
    MUJOCO_MENAGERIE_PATH=/workspace/third_party/mujoco_menagerie \
    PIP_CACHE_DIR=/tmp/pip-cache \
    HOME=/tmp/home

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglfw3 \
    libgles2 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    patchelf \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml setup.py README.md requirements.txt /workspace/
COPY genedynamics /workspace/genedynamics
COPY configs /workspace/configs
COPY scripts /workspace/scripts
COPY docker/install /opt/genedynamics-docker/install

RUN --mount=type=cache,target=/tmp/pip-cache \
    bash /opt/genedynamics-docker/install/install_cpu_dev.sh /workspace

CMD ["/bin/bash"]


FROM base-cpu AS dev-cpu

ENV JAX_PLATFORMS=cpu


FROM base-cpu AS sim-cpu

ENV JAX_PLATFORMS=cpu \
    MUJOCO_GL=osmesa


FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS base-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    MUJOCO_MENAGERIE_PATH=/workspace/third_party/mujoco_menagerie \
    PIP_CACHE_DIR=/tmp/pip-cache \
    HOME=/tmp/home

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglfw3 \
    libgles2 \
    libsm6 \
    libxext6 \
    libxrender1 \
    patchelf \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

WORKDIR /workspace

COPY pyproject.toml setup.py README.md requirements.txt /workspace/
COPY genedynamics /workspace/genedynamics
COPY configs /workspace/configs
COPY scripts /workspace/scripts
COPY docker/install /opt/genedynamics-docker/install

RUN --mount=type=cache,target=/tmp/pip-cache \
    bash /opt/genedynamics-docker/install/install_gpu_train.sh /workspace

CMD ["/bin/bash"]


FROM base-gpu AS train-gpu

ENV JAX_PLATFORMS=cuda,cpu \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
