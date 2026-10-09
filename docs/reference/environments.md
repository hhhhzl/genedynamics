# Environments

V1 includes six task families, exposed through seven configuration keys. Choose
a task, then select a planner from the [planner catalogue](planners.md).

| Task | What it provides | Starting configuration | Implementation |
| --- | --- | --- | --- |
| **Planar navigation** | A bounded 2D point robot for constrained planning and obstacle avoidance. | [MD-COAS](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/configs/single_2d/mdcoas.yaml) | [Single integrator](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/envs/domains/toy/single_integrator_box_2d.py) |
| **7-DoF arm avoidance** | D3IL obstacle avoidance with Cartesian XY or seven-joint velocity actions. | [MD-COAS](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/configs/d3il_avoiding/mdcoas.yaml) | [Cartesian interface](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/envs/external/d3il/avoiding_env.py) · [Joint-velocity interface](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/envs/external/d3il/avoiding_env_7d_vel.py) |
| **Quadruped stepping stones** | Joint planning of body motion, foot residuals and gait phase over discrete footholds. | [2GO](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/configs/quadruped/stepping_stones_2d/main/twogo.yaml) | [Stepping-stone model](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/envs/domains/quadruped/stepping_stones.py) |
| **Humanoid corridor** | Navigation through narrow passages using body pose, height and arm posture. | [2GO · Zone A](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/configs/humanoid/corridor_2d/main/twogo_zone_a.yaml) | [Corridor model](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/envs/domains/humanoid/corridor.py) |
| **Surface scanning** | Contact scanning across planar, cylindrical and NURBS surfaces with motion, stiffness and force control. | [MGA](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/configs/arm/surface_scan/main/mga.yaml) | [Surface-scan API](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/envs/domains/manipulation/surface_scan_brax.py) |
| **Peg insertion** | Contact-rich insertion with motion–impedance actions and pose, clearance, friction and sensing variants. | [MGA](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/configs/arm/peg_insert/main/mga.yaml) | [Insertion environment](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/envs/domains/manipulation/peg_insert_brax.py) |

## Configuration keys

| `env_name` | Interface |
| --- | --- |
| `single_integrator_box_2d` | 2D position state and velocity action. |
| `d3il_avoiding` | Cartesian interface: 4D state containing commanded and measured XY; 2D XY action. |
| `d3il_avoiding_9d` | Joint interface: 9D state containing TCP XY and seven joint positions; 7D joint-velocity action. |
| `quadruped_stepping_stones_2d` | Reduced planning model with 16D state and 12D action. |
| `humanoid_corridor_2d` | Reduced planning model with 14D state and 9D action. |
| `manipulator_surface_scan` | MuJoCo/MJX contact simulation with a motion–stiffness–force primitive. |
| `manipulator_peg_insert` | MuJoCo/MJX contact simulation with a pose–stiffness–force primitive. |

These are experiment configuration keys. D3IL tasks are constructed through
the runner's environment plugins; they are not a promise that every key can be
passed directly to `make_env()`.

## Run a task

Start with the [planar CPU recipe](../recipes/planar-cpu.md), the
[humanoid corridor recipe](../recipes/humanoid-corridor.md), or the
[algorithm recipes](../recipes/index.md). Inspect a configuration before running:

```bash
genedynamics-run configs/single_2d/mdcoas.yaml \
  --device cpu --seed 0 --level 0 --dry-run
```

Install the [optimization and simulation extras](../getting-started/installation.md)
required by the selected recipe. D3IL also requires its
[integration setup](../integrations/d3il.md). MuJoCo, MJX and Brax are simulation
layers; they do not introduce additional task families.

The [V1 compatibility contract](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/release/v1_compatibility.json)
defines the release scope. Additional source implementations are available for
experimentation, but are outside this supported task catalogue. Humanoid
walking/push execution and MGA GPU qualification remain follow-up work.
