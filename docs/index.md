# GenerativeDynamics

**Generative models for robot learning, planning, and control.**

![MDOC, MD-COAS, 2GO, MGA, and real robot demonstrations](assets/showcase.gif)

Train reusable policy priors, learn trajectory diffusion models, and perform
model-based generative inference with dynamics and constraints. Connect the
resulting plans to closed-loop control through shared task, solver and execution
interfaces.

[Watch the full-quality video](assets/showcase.mp4) ·
[Explore the task recipes](recipes/index.md)

## 🧩 Build with reusable components

| Feature | What you can use |
| --- | --- |
| **🧠 Robot learning** | PPO/SAC training and checkpoint reuse; PPO horizon priors for MGA. |
| **✨ Generative models** | Learned trajectory diffusion, model-based generative inference and composable reverse transports. |
| **Planner plugins** | MDOC, MD-COAS, 2GO, MGA and comparison algorithms through shared task and solver interfaces. |
| **Geometry and constraints** | Convex primitives, meshes, signed-distance geometry, collision constraints, state/action limits and constraint schedules. |
| **Robot task composition** | Semantic robot profiles, task objectives, dynamics and scene geometry kept separate from planner implementations. |
| **Simulation adapters** | MuJoCo, MJX, Brax and D3IL integrations with recipe-specific dependencies. |
| **Execution components** | Receding-horizon planning, followers, controllers, governors, safety filters and robot I/O. |
| **Experiment tooling** | Seeded configurations, resumable task matrices, metrics, trajectories, reports and observer plugins. |
| **Compute** | JAX-first planning with CPU setup and CUDA installation paths; device support is qualified per planner. |

## 🧪 Explore our algorithms

| Algorithm | Start building |
| --- | --- |
| **MDOC** | Run the planar planning component with `configs/single_2d/mdoc.yaml`; see the [original multi-robot project](https://github.com/hhhhzl/mdoc) for its CBS application. |
| **MD-COAS** | Explore constrained diffusion and adaptive scheduling with `configs/single_2d/mdcoas.yaml`, then D3IL arm avoidance with `configs/d3il_avoiding/mdcoas.yaml`. |
| **2GO** | Plan quadruped stepping stones and humanoid corridor motion; begin with the [humanoid corridor recipe](recipes/humanoid-corridor.md). |
| **MGA** | Reuse learned policy priors for generative motion–impedance control in [peg insertion and surface scanning](recipes/index.md#mga). |

The [algorithm catalogue](reference/planners.md) links the papers, explains each
planner and maps published names to configuration keys.

## 🌍 Environments and constraints

Browse six [environment families](reference/environments.md) and the
[constraint catalogue](reference/constraints.md): CBF/CFS filters, adaptive
scheduling, and closed-form, JAXopt OSQP, OSQP and CVXOPT numerical solvers.

## 🚀 Start here

| Goal | Guide |
| --- | --- |
| Install the package and optional simulators | [Installation](getting-started/installation.md) |
| Use an environment and planner in your own Python loop | [Direct Python example](getting-started/python-api.md) |
| Save trajectories, metrics and multi-seed experiments | [First experiment](getting-started/quickstart.md) |
| Train models and reuse learned priors | [Learning workflows](guides/learning-and-priors.md) |
| Choose a paper algorithm and task | [Recipes](recipes/index.md) |
| Understand or extend the interfaces | [Architecture](concepts/architecture.md) · [Plugin development](guides/adding-a-plugin.md) |
| Check support or upcoming integrations | [Compatibility](reference/compatibility.md) · [Roadmap](reference/roadmap.md) |

![Platform architecture](assets/architecture.svg)

## 📍 Release status

V1 is JAX-first and includes six task families. Dedicated Torch integrations
support DPCC and SafeDiffuser; broader Torch support and Rust/C++ backends are
planned. MGA GPU qualification and humanoid walking/push are follow-up
priorities. See the [roadmap](reference/roadmap.md), [compatibility matrix](reference/compatibility.md) and
[V1 scope](releases/v1_scope.md) for the current release boundaries.
