<h1 align="center">GenerativeDynamics</h1>

<p align="center">
  <strong>A modular engine for robot planning and control.</strong><br>
  Generate trajectories. Compose constraints. Connect to control.
</p>

<p align="center">
  <a href="docs/index.md">Documentation</a> ·
  <a href="#get-started">Get started</a> ·
  <a href="#explore-the-examples">Examples</a> ·
  <a href="#how-the-stack-fits-together">Architecture</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

<p align="center">
  <a href="docs/getting-started/installation.md"><img src="https://img.shields.io/badge/Python-3.10–3.12-334155?logo=python&logoColor=white" alt="Python 3.10 to 3.12"></a>
  <a href="docs/reference/compatibility.md"><img src="https://img.shields.io/badge/Compute-JAX-334155" alt="JAX compute"></a>
  <a href="docs/releases/v1_scope.md"><img src="https://img.shields.io/badge/Release-0.1.0_alpha-334155" alt="0.1.0 alpha release candidate"></a>
</p>

<p align="center">
  <img src="docs/assets/showcase.gif" width="100%" alt="A continuous gallery of native MDOC, MD-COAS, 2GO and MGA demonstrations, moving from planning and simulation to real robots.">
</p>

<p align="center">
  <a href="docs/assets/showcase.mp4">Full-quality video</a> ·
  <a href="docs/assets/README.md">Demonstrations and sources</a> ·
  <a href="#explore-the-examples">Run the examples</a>
</p>

GenerativeDynamics connects **trajectory generation, geometric constraints, and
closed-loop execution** in one robotics stack. Build a task once, compare
planners on the same problem, and compose the controllers, safety filters and
robot adapters around it. Use individual components directly in Python or run
complete experiments from a configuration.

## Updates

- **2026-10-09** — Added a direct Python planning example, a unified paper and
  hardware gallery, and a paper-linked planner catalogue.
- **2026-10-05** — Prepared the V1 release candidate with six task families,
  optional integrations, user documentation and CPU validation evidence.

Follow [CHANGELOG.md](CHANGELOG.md) for new environments, planners and integrations.

## Built to compose

| Feature | What you get |
| --- | --- |
| **Planning** | Diffusion trajectory generation, constrained optimization, sampling MPC and contact-aware motion–impedance control. |
| **Tasks and robots** | Planar navigation, 7-DoF avoidance, quadruped footholds, humanoid corridors, peg insertion and surface scanning. |
| **Geometry and constraints** | Convex primitives, meshes, signed-distance geometry and CSG; composable collision, state, action and contact constraints. |
| **Simulation adapters** | MuJoCo, MJX, Brax and D3IL integrations, with physics and task adapters separate from solver logic. |
| **Compute** | JAX planning and batched rollouts; CPU and CUDA installation paths. Dedicated Torch integrations for DPCC and SafeDiffuser. |
| **Execution components** | Robot I/O, controllers, trajectory followers, governors, safety filters, recovery policies and observers. |
| **Experiment tooling** | YAML task matrices, seeds, resumable runs, resolved configurations, metrics, traces, replay and reports. |
| **Extension points** | Registered planners, environments, geometry, metrics and visualizations; optional integrations with explicit dependencies. |

See [device and integration support](docs/reference/compatibility.md) for the
qualified paths, and [architecture](docs/concepts/architecture.md) for the
interfaces behind these features.

## Get started

Create a planar environment, give it a **2GO** planner, and move from `(0.8, 0.8)`
to the origin. This example runs on CPU without a simulator or model checkpoint.

```bash
git clone https://github.com/hhhhzl/genedynamics.git
cd genedynamics
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[optimization]"
```

```python
import jax
import numpy as np

from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.envs import make_env, make_energy
from genedynamics.solvers import TwoGOSolver

RuntimeBackendManager.set_backend("jax", device="cpu")
env = make_env("single_integrator_box_2d")
planner = TwoGOSolver(
    dynamics=env,
    energy=make_energy("single_integrator_box_2d"),
    backend=RuntimeBackendManager.get_backend(),
    dt=env.dt,
    horizon=20,
    Nsample=128,
    Ndiffuse=16,
)

state = np.array([0.8, 0.8], dtype=np.float32)
for step in range(60):
    key = jax.random.fold_in(jax.random.PRNGKey(0), step)
    plan = planner.solve(state, horizon=20, rng_key=key)
    state = env.transition(state, plan.actions[0])
    if np.linalg.norm(state) < 0.1:
        break

print(f"Goal error: {np.linalg.norm(state):.3f}")
```

The CPU validation reaches the goal in 48 steps, with `Goal error: 0.099`.
Run the complete example with `python examples/plan_to_goal.py`, or follow the
[Python API walkthrough](docs/getting-started/python-api.md) to adapt it.

For configured experiments, the [first-run guide](docs/getting-started/quickstart.md)
covers result artifacts and resuming runs. The [installation guide](docs/getting-started/installation.md)
covers CUDA, simulator assets, learned planners and deployment extras.

## Explore the examples

Start with one of our four methods. Each recipe connects an algorithm to a task,
its dependencies and an inspectable configuration.

| Method | Explore | Starting configurations |
| --- | --- | --- |
| **[MDOC](https://arxiv.org/abs/2607.12423)** | Diffusion planning with safety projections in constrained scenes. | [Planar navigation](configs/single_2d/mdoc.yaml) |
| **[MD-COAS](https://arxiv.org/abs/2607.14455)** | Constraint-guided diffusion, from planar obstacles to 7-DoF avoidance. | [Planar navigation](configs/single_2d/mdcoas.yaml) · [Arm avoidance](configs/d3il_avoiding/mdcoas.yaml) |
| **[2GO](https://arxiv.org/abs/2610.07772)** | Geometry-guided trajectory optimization for constrained locomotion. | [Quadruped stepping stones](configs/quadruped/stepping_stones_2d/main/twogo.yaml) · [Humanoid corridor](configs/humanoid/corridor_2d/main/twogo_zone_c.yaml) |
| **MGA** | Generative motion–impedance control for contact-rich tasks. | [Peg insertion](configs/arm/peg_insert/main/mga.yaml) · [Surface scanning](configs/arm/surface_scan/main/mga.yaml) |

The [recipe catalogue](docs/recipes/index.md) explains setup and assets. The
homepage also includes original paper demonstrations: MDOC multi-robot
coordination is available in its [original project](https://github.com/hhhhzl/mdoc),
and MGA humanoid contact footage previews research beyond the qualified V1
recipes. See the [demo map](docs/assets/README.md) for the distinction.

## Included planners and baselines

**Ours** marks our algorithms. Each entry links its original paper or current
implementation; configuration variants are documented in the
[planner reference](docs/reference/planners.md).

| Algorithm | What it does | Source |
| --- | --- | --- |
| **MDOC · Ours** | Model-based diffusion with control-barrier-function projections inside dynamics rollouts. | [Paper](https://arxiv.org/abs/2607.12423) · [Original multi-robot project](https://github.com/hhhhzl/mdoc) |
| **MD-COAS · Ours** | Combines an augmented-Lagrangian feasibility prior, CFS projection, and adaptive constraint scheduling. | [Paper](https://arxiv.org/abs/2607.14455) |
| **2GO · Ours** | Shapes generative trajectory updates and exploration with active constraint geometry for constrained locomotion. | [Paper](https://arxiv.org/abs/2610.07772) |
| **MGA · Ours** | Combines RL sequence proposals, model-based evaluation, and realization-aware geometry for motion–impedance control. | [Implementation](https://github.com/hhhhzl/genedynamics/tree/release/v1-open-source/genedynamics/solvers/single/mga) |
| **MBD** | Uses known dynamics and Monte Carlo score estimates to optimize trajectories without demonstrations. | [Paper](https://arxiv.org/abs/2407.01573) · [Project](https://lecar-lab.github.io/mbd/) |
| **EB-MBD** | Introduces emerging barriers during model-based diffusion to retain useful samples under constraints. | [Paper](https://arxiv.org/abs/2510.07700) |
| **MPPI** | Updates sampled control sequences using exponential rollout-cost weights. | [Paper](https://arxiv.org/abs/1707.02342) |
| **DIAL-MPC** | Uses diffusion-inspired annealing to refine sampled controls in receding-horizon MPC. | [Paper](https://arxiv.org/abs/2409.15610) |
| **ATACOM** | Learns and executes actions in the tangent space of an augmented constraint manifold. | [Paper](https://proceedings.mlr.press/v164/liu22c.html) |
| **ISSA** | Filters proposed actions through a derivative-free implicit-safe-set search. | [Paper](https://proceedings.mlr.press/v164/zhao22a.html) |
| **PegasusFlow** | Uses rolling denoising and weighted spline-basis updates for model-based trajectory optimization. | [Paper](https://arxiv.org/abs/2509.08435) |
| **DPCC** | Adds model-based projections and constraint tightening to learned trajectory diffusion. | [Paper](https://proceedings.mlr.press/v283/romer25a.html) |
| **SafeDiffuser** | Applies control-barrier-function corrections within learned diffusion denoising. | [Paper](https://arxiv.org/abs/2306.00148) |

**Naming:** CFS-MBD is the historical implementation name for **MD-COAS**.
Learned integrations require their optional dependencies and
checkpoints. See [planner setup and implementation notes](docs/reference/planners.md).

## How the stack fits together

<img src="docs/assets/architecture.svg" width="100%" alt="Tasks, planners, and constraints compose through a shared planning and control runtime. Environment adapters connect to simulation, and control components connect to execution. Metrics, traces, and visualization span both paths.">

**Tasks** define dynamics, objectives and geometry. **Planners** produce
trajectories. **Execution** connects plans to followers, controllers, safety
filters and robot I/O. **Observers and metrics** record the result across the
stack. These interfaces let an application replace a solver or add a robot
while reusing its task and surrounding tools.

Read the [architecture guide](docs/concepts/architecture.md),
[configure a task](docs/guides/configuration.md), or
[add a plugin](docs/guides/adding-a-plugin.md).

## Roadmap

| Direction | Planned capability |
| --- | --- |
| **Portable compute** | Qualify MGA on CUDA; expand Torch planner parity and device-resident transfers; evaluate MuJoCo Warp rollouts. Scope Rust/C++ first to control and I/O components. |
| **Robot deployment** | Add reviewed Panda and xArm7 hardware recipes, joint-trajectory and Cartesian impedance controllers, and `ros2_control` execution alongside existing G1 components. |
| **Predictable execution** | Decouple planning and control rates; standardize command expiry, cancellation and fallback; measure jitter, deadline misses and long-run reliability. |
| **Learned-policy integration** | Connect action-chunk and VLA policies through proposal/warm-start adapters; explore OpenPI inference and LeRobot recording/replay. |
| **Live scenes** | Extend frame-aware geometry with timestamped perception and moving obstacles for calibrated manipulation. |
| **Reproducible evaluation** | Publish hardware-specific latency, throughput, memory, task-quality and safety results, plus replayable benchmark artifacts and optional plugins. |

These are development directions. The [detailed roadmap](docs/reference/roadmap.md)
connects each one to existing interfaces and measurable release gates.

V1 (`0.1.0`) is a release candidate with an evolving public API. MGA GPU
qualification and humanoid walking/push remain follow-up work. Current
[compatibility](docs/reference/compatibility.md), [validation evidence](docs/releases/v1_evidence.md)
and [benchmark methodology](docs/reference/metrics.md) define what is supported
and measured today.

## Documentation

| Start building | Go deeper |
| --- | --- |
| [Installation](docs/getting-started/installation.md) | [Architecture](docs/concepts/architecture.md) |
| [Python API example](docs/getting-started/python-api.md) | [Planners and sources](docs/reference/planners.md) |
| [Configured experiments](docs/getting-started/quickstart.md) | [Configuration](docs/guides/configuration.md) |
| [Task recipes](docs/recipes/index.md) | [Plugin development](docs/guides/adding-a-plugin.md) |
| [D3IL integration](docs/integrations/d3il.md) | [API reference](docs/api/README.md) |

## Contributing

Bring a planner, a robot task, an integration or a documentation improvement.
Start with [CONTRIBUTING.md](CONTRIBUTING.md). Community guidelines are in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md); vulnerability reporting is in
[SECURITY.md](SECURITY.md).

## Research and citation

The solvers build on research algorithms; the framework provides the interfaces
and runtime around them. Use [CITATION.cff](CITATION.cff) when citing the project,
and the papers linked above when using individual methods.

## License

License selection is pending for the public V1 release. The
[release checklist](docs/releases/v1_checklist.md) tracks this requirement.
