# Docker

Use an existing development image to run the checked-in examples. From the
repository root, mount the checkout at `/workspace`:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace \
  -e JAX_PLATFORMS=cpu genedynamics/dev-cpu:local \
  python examples/plan_to_goal.py --device cpu
```

On Linux with NVIDIA drivers, the NVIDIA Container Toolkit and the existing
GPU image:

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
  -e JAX_PLATFORMS=cuda genedynamics/train-gpu:local \
  python examples/plan_to_goal.py --device gpu --samples 1024
```

Configured runs save results back to the host's `results/` directory. CPU
rendering uses OSMesa and GPU rendering uses EGL. The image architecture and
installed dependencies must match the host and V1 numerical stack.

See the [complete Docker guide](https://github.com/hhhhzl/genedynamics/blob/main/docker/README.md)
for image checks, batched constraints, file ownership, headless experiments
and robot assets. This usage update reuses existing images and does not require
rebuilding them. See [GPU planning](../recipes/gpu-planning.md) for the task recipes.
