# Plan and execute from Python

Use an environment and a planner directly when you want to embed planning in
your own application. This example drives a two-dimensional point robot from
`[0.8, 0.8]` to the origin using 2GO on CPU. It needs no robot simulator,
checkpoint or result directory.

After [installing the package](installation.md), run the checked-in example
from the repository root:

```bash
python examples/plan_to_goal.py
```

On a configured NVIDIA host, run the same loop with a larger candidate batch:

```bash
JAX_PLATFORMS=cuda python examples/plan_to_goal.py --device gpu --samples 1024
```

The script checks the requested device before constructing the planner and
verifies backend array placement. See [GPU setup and recipes](../recipes/gpu-planning.md)
or [Docker](docker.md) for installation. Use `device="gpu"` in the runtime
selection below when embedding this loop in a CUDA application.

## Create the task and planner

The environment defines the dynamics and the energy defines the objective.
Select the runtime before constructing the planner:

```python
import jax
import numpy as np

from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.envs import make_env, make_energy
from genedynamics.solvers import TwoGOSolver

RuntimeBackendManager.set_backend("jax", device="cpu")
env_name = "single_integrator_box_2d"
env = make_env(env_name)
planner = TwoGOSolver(
    dynamics=env,
    energy=make_energy(env_name),
    backend=RuntimeBackendManager.get_backend(),
    dt=env.dt,
    horizon=20,
    Nsample=128,
    Ndiffuse=16,
)
```

`horizon` is the number of future control steps in each plan. `Nsample`
controls the candidate batch size, and `Ndiffuse` controls the number of
generative updates. Matching `dt` keeps planning and execution on the same
time step.

## Execute the first action and replan

```python
state = np.array([0.8, 0.8], dtype=np.float32)
for step in range(60):
    key = jax.random.fold_in(jax.random.PRNGKey(0), step)
    plan = planner.solve(state, horizon=20, rng_key=key)
    state = env.transition(state, plan.actions[0])
    if np.linalg.norm(state) < 0.1:
        break

print(f"Goal error: {np.linalg.norm(state):.3f}")
```

`solve` returns a `Trajectory` containing states, actions and planner
diagnostics in `info`. Each loop executes the first action from the new plan,
then plans again from the resulting state. The explicit random keys make the
sequence repeatable within the selected numerical environment.

## Success criterion

The task succeeds when the Euclidean distance to the origin is below `0.1`,
within 60 executed steps. The checked-in example also asserts that the planned
states, planned actions and executed states are finite, and fails if the goal
is not reached.

A verified CPU run with JAX 0.6.2 produced:

```text
Reached goal in 48 steps (error=0.099).
```

This checks a complete planning/execution loop, rather than only testing that
the planner can return an array. The first call includes JAX compilation;
this small example is not a latency benchmark.

For obstacle-rich or robot tasks, start with the [algorithm recipes](../recipes/index.md),
which assemble the appropriate geometry, constraint filters and assets. Use
the [experiment runner](quickstart.md) when you need configuration files,
seed matrices, metrics and saved artifacts. See the
[planner catalogue](../reference/planners.md) for other algorithms and
[architecture](../concepts/architecture.md) for execution interfaces.
