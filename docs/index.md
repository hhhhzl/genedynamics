# GenerativeDynamics

**A modular engine for robot planning and control.**

![MDOC, MD-COAS, 2GO, MGA, and real robot demonstrations](assets/showcase.gif)

Compose trajectory generation, geometric constraints, simulation, and
closed-loop execution in one robotics stack. Start with a task recipe, choose a
planner, and connect the components your application needs.

[Watch the full-quality video](assets/showcase.mp4) ·
[Explore the task recipes](recipes/index.md)

## Build with reusable components

| Feature | What you can use |
| --- | --- |
| **Planner plugins** | MDOC, MD-COAS, 2GO, MGA and comparison algorithms through shared task and solver interfaces. |
| **Geometry and constraints** | Convex primitives, meshes, signed-distance geometry, collision constraints, state/action limits and constraint schedules. |
| **Robot task composition** | Semantic robot profiles, task objectives, dynamics and scene geometry kept separate from planner implementations. |
| **Simulation adapters** | MuJoCo, MJX, Brax and D3IL integrations with recipe-specific dependencies. |
| **Execution components** | Receding-horizon planning, followers, controllers, governors, safety filters and robot I/O. |
| **Policy proposals** | Shared prior interfaces for RL and learned-diffusion proposals and warm starts. |
| **Experiment tooling** | Seeded configurations, resumable task matrices, metrics, trajectories, reports and observer plugins. |
| **Compute** | JAX-first planning with CPU setup and CUDA installation paths; device support is qualified per planner. |

## Explore our algorithms

| Algorithm | Start building |
| --- | --- |
| **MDOC** | Run the planar planning component with `configs/single_2d/mdoc.yaml`; see the [original multi-robot project](https://github.com/hhhhzl/mdoc) for its CBS application. |
| **MD-COAS** | Explore constrained diffusion and adaptive scheduling with `configs/single_2d/mdcoas.yaml`, then D3IL arm avoidance with `configs/d3il_avoiding/mdcoas.yaml`. |
| **2GO** | Plan quadruped stepping stones and humanoid corridor motion; begin with the [humanoid corridor recipe](recipes/humanoid-corridor.md). |
| **MGA** | Run contact-rich peg insertion or compliant surface scanning with the [MGA recipes](recipes/index.md#mga). |

The [algorithm catalogue](reference/planners.md) links the papers, explains each
planner and maps published names to configuration keys.

## Start here

| Goal | Guide |
| --- | --- |
| Install the package and optional simulators | [Installation](getting-started/installation.md) |
| Use an environment and planner in your own Python loop | [Direct Python example](getting-started/python-api.md) |
| Save trajectories, metrics and multi-seed experiments | [First experiment](getting-started/quickstart.md) |
| Choose a paper algorithm and task | [Recipes](recipes/index.md) |
| Understand or extend the interfaces | [Architecture](concepts/architecture.md) · [Plugin development](guides/adding-a-plugin.md) |
| Check support or upcoming integrations | [Compatibility](reference/compatibility.md) · [Roadmap](reference/roadmap.md) |

![Platform architecture](assets/architecture.svg)

## Release status

V1 is JAX-first and includes six task families. Dedicated Torch integrations
support DPCC and SafeDiffuser; broader Torch support and Rust/C++ backends are
planned. MGA GPU qualification and humanoid walking/push are follow-up
priorities. See the [roadmap](reference/roadmap.md), [compatibility matrix](reference/compatibility.md) and
[V1 scope](releases/v1_scope.md) for the current release boundaries.
