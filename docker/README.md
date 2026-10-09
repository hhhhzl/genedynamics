# Run with Docker

Build locally from the repository root. Images include the V1 package,
examples, configurations and simulation dependencies. Compose mounts the
checkout at `/workspace`, so saved results remain on the host and code edits
are visible immediately.

| Target | Host | Rendering | Use |
| --- | --- | --- | --- |
| `dev-cpu` / `sim-cpu` | Linux or Docker Desktop, native amd64/arm64 | OSMesa | Planning, development and headless simulation |
| `train-gpu` | Linux amd64 + NVIDIA | EGL | JAX CUDA planning and optional Torch training |

## CPU: build and run

```bash
docker compose -f docker/compose.cpu.yml build
docker compose -f docker/compose.cpu.yml run --rm genedynamics-dev-cpu \
  python examples/plan_to_goal.py --device cpu
docker compose -f docker/compose.cpu.yml run --rm genedynamics-dev-cpu \
  python examples/batched_constraints.py --device cpu
```

CPU Compose uses the host's architecture, including Apple Silicon. No local
Python installation or custom environment file is needed. On Linux, match
output ownership before running Compose:

```bash
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
```

Run a configured MD-COAS experiment and retain its artifacts:

```bash
docker compose -f docker/compose.cpu.yml run --rm genedynamics-dev-cpu \
  genedynamics-run configs/single_2d/mdcoas.yaml \
  --device cpu --seed 0 --level 6 \
  --development-root results/_development/docker-mdcoas
```

The output appears in `results/_development/docker-mdcoas` on the host.
Append `--dry-run` to validate configuration before allocating a run. Start
an interactive shell by omitting the final command:

```bash
docker compose -f docker/compose.cpu.yml run --rm genedynamics-dev-cpu
```

## GPU: Linux + NVIDIA

Install the host NVIDIA driver, Docker Engine with Compose 2.30 or newer,
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Compose requests GPUs using its [GPU service support](https://docs.docker.com/compose/how-tos/gpu-support/).
Docker Desktop on macOS does not provide this NVIDIA CUDA path.

```bash
docker compose -f docker/compose.gpu.yml build
docker compose -f docker/compose.gpu.yml run --rm genedynamics-train-gpu nvidia-smi
docker compose -f docker/compose.gpu.yml run --rm genedynamics-train-gpu \
  python -c "import jax; print(jax.devices('gpu'))"
docker compose -f docker/compose.gpu.yml run --rm genedynamics-train-gpu \
  python examples/plan_to_goal.py --device gpu --samples 1024
docker compose -f docker/compose.gpu.yml run --rm genedynamics-train-gpu \
  python examples/batched_constraints.py --device gpu --samples 4096
```

The GPU service sets `JAX_PLATFORMS=cuda`; initialization fails if CUDA is
unavailable. Use `--device gpu` for the framework's JAX paths. Select one GPU
with `run --rm -e CUDA_VISIBLE_DEVICES=0 ...`. See the
[GPU recipes](../docs/recipes/gpu-planning.md) for MDOC, MD-COAS and robot tasks.
MGA GPU qualification remains deferred.

## Headless robot simulation

The image contains packaged robot assets. Recipes that use Menagerie must
initialize the pinned assets in the mounted checkout:

```bash
docker compose -f docker/compose.cpu.yml run --rm genedynamics-dev-cpu \
  bash scripts/setup/setup_mujoco_menagerie.sh
```

CPU uses `MUJOCO_GL=osmesa`; GPU uses `MUJOCO_GL=egl`. These services save
rendered artifacts rather than opening a desktop viewer. D3IL additionally
needs its [integration setup](../docs/integrations/d3il.md), including
Pinocchio and task data; a CUDA image alone does not install those assets.

## Versions and verification

Both installers pin JAX 0.6.2, Brax 0.14.1 and MuJoCo/MJX 3.6.0 and run
`pip check`. Other dependencies follow the package extras; these are
rebuildable development images, not a fully locked production environment.
The GPU image also pins Torch 2.10.0 / torchvision 0.25.0 with CUDA 12.6 wheels.
Images are built locally; no GitHub Packages image is required.

A successful device check proves visibility. Task success, numerical parity,
warm latency and GPU memory still require measurements on the target host.
The [release evidence](../docs/releases/v1_evidence.md) records what was
actually validated. The separate `isaac-lab.Dockerfile` is experimental and
outside the V1 container qualification.
