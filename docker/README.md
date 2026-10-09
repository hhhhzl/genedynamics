# Use the development containers

Run GPU and CPU examples with the existing local development images. Mount
the repository at `/workspace` so the examples and current code are available,
and experiment results remain on the host.

| Image | Host | Rendering |
| --- | --- | --- |
| `genedynamics/dev-cpu:local` | CPU host matching the built image architecture | OSMesa |
| `genedynamics/train-gpu:local` | Linux amd64 + NVIDIA | EGL |

## Check the existing images

From the repository root:

```bash
docker image inspect genedynamics/dev-cpu:local
```

GPU users check `genedynamics/train-gpu:local` instead. These commands need
an already built local image; no registry image is published by this guide.
The image must contain the V1 dependency stack, including JAX 0.6.2.

## CPU examples

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace \
  -e JAX_PLATFORMS=cpu genedynamics/dev-cpu:local \
  python examples/plan_to_goal.py --device cpu

docker run --rm -v "$PWD:/workspace" -w /workspace \
  -e JAX_PLATFORMS=cpu genedynamics/dev-cpu:local \
  python examples/batched_constraints.py --device cpu --samples 4096
```

Inspect the numerical stack without changing the image:

```bash
docker run --rm genedynamics/dev-cpu:local \
  python -c "import jax, numpy; print(jax.__version__, numpy.__version__, jax.devices())"
```

## NVIDIA GPU examples

Use Linux with the host NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Docker Desktop on macOS does not provide this NVIDIA CUDA path.

```bash
docker run --rm --gpus all genedynamics/train-gpu:local nvidia-smi

docker run --rm --gpus all -e JAX_PLATFORMS=cuda \
  genedynamics/train-gpu:local \
  python -c "import jax; print(jax.devices('gpu'))"

docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
  -e JAX_PLATFORMS=cuda genedynamics/train-gpu:local \
  python examples/plan_to_goal.py --device gpu --samples 1024

docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
  -e JAX_PLATFORMS=cuda genedynamics/train-gpu:local \
  python examples/batched_constraints.py --device gpu --samples 4096
```

`JAX_PLATFORMS=cuda` makes missing CUDA initialization an error. The examples
also check actual array placement. Select one GPU by adding
`-e CUDA_VISIBLE_DEVICES=0`. See [GPU recipes](../docs/recipes/gpu-planning.md)
for MDOC, MD-COAS and 2GO configurations. MGA GPU remains deferred.

## Save a configured experiment

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace \
  -e JAX_PLATFORMS=cpu -e MUJOCO_GL=osmesa genedynamics/dev-cpu:local \
  genedynamics-run configs/single_2d/mdcoas.yaml \
  --device cpu --seed 0 --level 6 \
  --development-root results/_development/docker-mdcoas
```

Artifacts appear in `results/_development/docker-mdcoas` on the host. Append
`--dry-run` to inspect configuration first. On Linux, add
`--user "$(id -u):$(id -g)"` when host-owned output files are needed.
For GPU experiments, use `--gpus all`, the GPU image, `JAX_PLATFORMS=cuda`,
`MUJOCO_GL=egl` and the runner's `--device gpu` flag.

## Robot assets and integrations

CPU rendering uses OSMesa; GPU rendering uses EGL. These settings support
headless artifact rendering rather than a desktop viewer. Menagerie-based
recipes need the pinned assets in the mounted checkout:

```bash
bash scripts/setup/setup_mujoco_menagerie.sh
```

D3IL requires its [integration setup](../docs/integrations/d3il.md), including
Pinocchio and task data. DPCC/SafeDiffuser additionally need the Torch extra
and trained checkpoints; image GPU visibility does not establish task readiness.

This guide reuses the existing images and Docker definitions. It does not
qualify a new image build, GPU numerical parity or deployment performance.
See [release evidence](../docs/releases/v1_evidence.md) for executed checks.
