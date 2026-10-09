# Docker

Run the examples without installing Python dependencies on your host. From
the repository root, build the CPU image and run a complete planning loop:

```bash
docker compose -f docker/compose.cpu.yml build
docker compose -f docker/compose.cpu.yml run --rm genedynamics-dev-cpu \
  python examples/plan_to_goal.py --device cpu
```

On Linux with NVIDIA drivers and the NVIDIA Container Toolkit:

```bash
docker compose -f docker/compose.gpu.yml build
docker compose -f docker/compose.gpu.yml run --rm genedynamics-train-gpu \
  python -c "import jax; print(jax.devices('gpu'))"
docker compose -f docker/compose.gpu.yml run --rm genedynamics-train-gpu \
  python examples/plan_to_goal.py --device gpu --samples 1024
```

Compose mounts the checkout at `/workspace`. Configured runs save results
back into the host's `results/` directory. CPU rendering uses OSMesa and GPU
rendering uses EGL. The CPU image uses the host's native architecture;
the NVIDIA image targets Linux amd64.

Follow the [complete Docker guide](https://github.com/hhhhzl/genedynamics/blob/main/docker/README.md)
for prerequisites, file ownership, GPU selection, saved experiments and robot
assets. See [GPU planning](../recipes/gpu-planning.md) for constrained tasks
and a batched constraint example.
