# GPU planning and batched constraints

Run the JAX planners on a Linux machine with an NVIDIA GPU. Start with a
complete 2GO control loop, then use the existing configurations for constrained
planning. These examples use one visible GPU; they do not distribute work
across multiple GPUs.

## Install and select the GPU

Use Python 3.10–3.12 in a virtual environment. From the repository root:

```bash
python -m pip install -e ".[optimization]" \
  -r requirements/gpu-jax.txt "jax[cuda12]==0.6.2"
JAX_PLATFORMS=cuda python -c "import jax; print(jax.devices('gpu'))"
```

The requirements file includes simulation dependencies. The explicit JAX
version keeps the examples on the V1 numerical stack. Check the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)
for NVIDIA driver and CUDA requirements; CUDA is unavailable on macOS.

Use `--device gpu` for these JAX commands. `JAX_PLATFORMS=cuda` makes a missing
CUDA backend an error. To select a particular physical GPU, prefix a command
with `CUDA_VISIBLE_DEVICES=0` and change `0` to the desired device index.

For an isolated CUDA environment, use the [GPU Docker service](../getting-started/docker.md).
It runs the same examples and keeps saved results in the mounted checkout.

## Plan, execute and replan with 2GO

```bash
JAX_PLATFORMS=cuda python examples/plan_to_goal.py \
  --device gpu --samples 1024
```

This simulator-free example constructs an environment and `TwoGOSolver`,
generates a plan, executes its first action and replans until the point robot
reaches the origin. It checks actual backend array placement before planning
and fails if the requested GPU is unavailable. `--samples` sets the candidate
count inside each refinement step. The success criterion is a goal error below
`0.1` within 60 executed steps; it is checked on the resulting trajectory.

Use `--device cpu --samples 128` for the original CPU smoke case. See the
[Python API walkthrough](../getting-started/python-api.md) for the application
loop and the distinction between `Nsample`, horizon and refinement steps.

## Run configured constrained tasks

These commands use one seed and a development output root. They keep the
configured algorithm, constraints and schedules together:

```bash
# MDOC: CBF correction during planar generative refinement.
JAX_PLATFORMS=cuda genedynamics-run configs/single_2d/mdoc.yaml \
  --device gpu --seed 0 --level 6 \
  --development-root results/_development/gpu-mdoc

# MD-COAS: full-horizon CFS with adaptive constraint scheduling.
JAX_PLATFORMS=cuda genedynamics-run configs/single_2d/mdcoas.yaml \
  --device gpu --seed 0 --level 6 \
  --development-root results/_development/gpu-mdcoas

# 2GO: quadruped foothold planning on stepping stones.
JAX_PLATFORMS=cuda genedynamics-run \
  configs/quadruped/stepping_stones_2d/main/twogo.yaml \
  --device gpu --seed 0 --level 1 \
  --development-root results/_development/gpu-stepping

# 2GO: humanoid corridor planning, Zone A.
JAX_PLATFORMS=cuda genedynamics-run \
  configs/humanoid/corridor_2d/main/twogo_zone_a.yaml \
  --device gpu --seed 0 --level 1 \
  --development-root results/_development/gpu-corridor
```

Append `--dry-run` to validate a configuration before execution. A dry run
checks configuration only; it does not initialize a GPU or test a planner.
Completed runs save the resolved configuration, trajectories, metrics and
execution status. Check feasibility and task success in those artifacts.
The stepping and corridor commands plan robot motion; their commands do not
start a deployment controller.

For seven-joint arm avoidance, complete the
[D3IL setup](../integrations/d3il.md), then run:

```bash
JAX_PLATFORMS=cuda genedynamics-run configs/d3il_avoiding/mdcoas.yaml \
  --device gpu --seed 0 --level 1 \
  --development-root results/_development/gpu-arm-avoidance
```

## Project a sample batch on the GPU

```bash
JAX_PLATFORMS=cuda python examples/batched_constraints.py \
  --device gpu --samples 4096 --horizon 20 --repeats 20
```

The example generates actions with shape `(4096, 20, 2)` and applies an existing
JAX halfspace projection helper through two `vmap` axes inside `jit`. It checks
device placement, finite outputs, action bounds and residuals, and reports the
first call separately from synchronized warm calls. The halfspaces are fixed
(`u[0] >= 0.2`, `u[1] >= -0.1`); this demonstrates batch composition, without
state-dependent trajectory dynamics. The helper uses finite projection
iterations rather than a general exact QP algorithm.

In the full planners, `Nsample` is the inner candidate axis. Supported filters
and rollouts map across this axis, while state propagation and successive
generative updates use separate JAX loops. `num_modes` is a separate output
axis, and `--seeds` selects runner entries; neither flag implies distributed
GPU execution. See [batched constraints](../guides/batched-constraints.md) for
how MDOC, MD-COAS and 2GO compose correction, rollout and generative inference.

## Device boundaries

| Path | GPU usage |
| --- | --- |
| JAX MDOC / MD-COAS / 2GO | JAX candidate evaluation and supported constraint kernels can execute on CUDA. Task setup, result serialization and rendering also use the host. |
| DPCC / SafeDiffuser | Specialized Torch inference requires a compatible Torch CUDA installation and the task's trained checkpoints. DPCC's SciPy projection is host-side; SafeDiffuser's enabled denoising QPs use batched Torch. Follow [learning and priors](../guides/learning-and-priors.md#learned-trajectory-diffusion). |
| OSQP / CVXOPT wrappers | CPU solver interfaces; choosing a JAX planner does not move these wrappers to CUDA. |
| MGA | V1 retains its CPU release path; CUDA qualification is deferred. |

The examples are runnable device-selection recipes. V1's recorded validation
is on CPU; GPU parity, latency, peak memory and repeated-run reliability must
be measured on the target hardware before using these recipes as deployment
evidence. A larger candidate batch can improve throughput utilization but also
increases memory use, especially when diffusion histories are retained.
