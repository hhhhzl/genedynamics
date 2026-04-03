# Docker Workflows

This directory provides a split Docker strategy for the current repo:

- `dev-cpu`
  For local development on Apple Silicon / CPU-only machines.
- `sim-cpu`
  For headless MuJoCo CPU simulation and rendering.
- `train-gpu`
  For Linux + NVIDIA training or MJX-heavy runs.

## Why the split exists

The repository currently mixes CPU, GPU, optional, and research dependencies in
[`requirements.txt`](/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/enerdynamics/requirements.txt)
and the existing
[`scripts/setup/setup.sh`](/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/enerdynamics/scripts/setup/setup.sh)
assumes a visible CUDA GPU.

That is not suitable for:

- Apple Silicon local development
- CPU-only MuJoCo rendering
- reproducible GPU training images

The Docker assets here avoid that by:

- baking only the dependency subset needed by each target
- installing the package with `pip install --no-deps -e /workspace`
- bind-mounting the repo at runtime so local edits appear immediately

## Files

- `Dockerfile`
  Multi-target image definition.
- `.env`
  Compose-level defaults for UID/GID interpolation.
- `compose.cpu.yml`
  Local Apple Silicon / CPU development flow.
- `compose.gpu.yml`
  Linux + NVIDIA training flow.
- `env/*.env.example`
  Example environment files.
- `install/*.sh`
  Target-specific dependency installers.

## CPU workflow for this Mac

Your machine is `arm64`, so the preferred local path is `linux/arm64` with the
`dev-cpu` target.

1. The default env files are already checked in:

```bash
cat docker/env/dev-cpu.env
cat docker/.env
```

2. If you want custom values, copy from the examples or edit the checked-in defaults:

```bash
cp docker/env/dev-cpu.env.example docker/env/dev-cpu.env
cp docker/.env.example docker/.env
```

3. If you want host-matching file ownership, edit [`docker/.env`](/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/enerdynamics/docker/.env).
On Docker Desktop for macOS the defaults are usually fine.

4. Build and start a shell:

```bash
docker compose -f docker/compose.cpu.yml build
docker compose -f docker/compose.cpu.yml run --rm genedynamics-dev-cpu
```

5. Inside the container, test MuJoCo import:

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -c "import jax; print(jax.default_backend())"
```

6. Run the corridor follower:

```bash
python3 scripts/tasks/robot/run_humanoid_corridor_g1_follower.py \
  --seed-dir results/humanoid/corridor_2d/smoke/twogo_zone_a/level_1/seed_0
```

## GPU workflow for Linux + NVIDIA

This target is for `linux/amd64` and requires:

- Docker Engine on Linux
- NVIDIA driver on the host
- NVIDIA Container Toolkit configured

1. The default env files are already checked in:

```bash
cat docker/env/train-gpu.env
cat docker/.env
```

2. If you want custom values, copy from the examples or edit the checked-in defaults:

```bash
cp docker/env/train-gpu.env.example docker/env/train-gpu.env
cp docker/.env.example docker/.env
```

3. If needed, edit [`docker/.env`](/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/enerdynamics/docker/.env)
to match the host UID/GID.

4. Build and start:

```bash
docker compose -f docker/compose.gpu.yml build
docker compose -f docker/compose.gpu.yml run --rm genedynamics-train-gpu
```

5. Validate GPU visibility:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## Notes

- The root `.dockerignore` excludes `third_party/` from image builds, so the
  image expects the repository to be bind-mounted at `/workspace`.
- MuJoCo render on CPU uses `MUJOCO_GL=osmesa`.
- GPU runs use `MUJOCO_GL=egl`.
- The images intentionally do not call `scripts/setup/setup.sh` because that
  script hard-codes a GPU-only installation path.

## Recommended next improvement

For a fully reproducible production build, split the repository dependencies
into:

- `requirements/base.txt`
- `requirements/sim.txt`
- `requirements/gpu.txt`
- `requirements/dev.txt`

Then update `setup.py` so package installation does not depend on the full
research stack.
