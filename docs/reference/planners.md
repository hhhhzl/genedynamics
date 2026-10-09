# Planners and algorithm sources

Choose a task configuration, then change its planner while keeping the task,
constraints, and evaluation tools. **Ours** identifies algorithms developed by
the GenerativeDynamics authors. The list below follows the algorithm names in
their papers; implementation keys are documented separately.

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

## Pick a configuration

| Algorithm | Example configuration | Compute path |
| --- | --- | --- |
| MDOC | `configs/single_2d/mdoc.yaml` | JAX |
| MD-COAS | `configs/single_2d/mdcoas.yaml` | JAX + optimization dependencies |
| 2GO | `configs/humanoid/corridor_2d/main/twogo_zone_a.yaml` | JAX + optimization dependencies |
| MGA | `configs/arm/peg_insert/main/mga.yaml` | JAX + simulation; learned prior assets depend on the recipe |
| MBD | `configs/single_2d/mbd.yaml` | JAX |
| EB-MBD | `configs/single_2d/ebmbd.yaml` | JAX |
| MPPI | `configs/humanoid/corridor_2d/main/mppi_zone_a.yaml` | JAX |
| DIAL-MPC | `configs/arm/peg_insert/baseline/dial.yaml` | JAX + simulation |
| ATACOM | `configs/arm/peg_insert/baseline/atacom.yaml` | JAX + simulation; trained tangent-action policy |
| ISSA | `configs/arm/peg_insert/baseline/issa.yaml` | JAX + simulation; trained nominal policy |
| PegasusFlow | `configs/arm/peg_insert/baseline/pegasusflow.yaml` | JAX + simulation |
| DPCC | `configs/d3il_avoiding/dpcc.yaml` | Optional Torch + D3IL; trained diffusion checkpoint |
| SafeDiffuser | `configs/d3il_avoiding/safediffuser.yaml` | Optional Torch + D3IL; trained diffusion checkpoint |

See [installation](../getting-started/installation.md) for extras and
[compatibility](compatibility.md) for the supported device paths. A paper link
identifies the source algorithm; it does not imply that every upstream
experiment, robot, or theoretical assumption is reproduced by this integration.

## Names used in code

- **MD-COAS** uses the historical `cfsmbd` / `cfsmbd_full` implementation keys.
  CFS-MBD is not a separate paper algorithm in this catalogue. The main
  `mdcoas.yaml` recipe enables adaptive scheduling; individual CFS configurations
  can select a different constraint projection or schedule.
- **2GO** uses the solver key `2go`; its task configuration filenames use
  `twogo`. These refer to the same algorithm.
- **DIAL-MPC** uses the solver key `dial`.
- **DPCC** trajectory-selection settings, task dimensions, and constraint
  variants configure one integration rather than introducing new planners.
- `d3il_unified` is an environment/solver execution adapter. It selects the
  algorithm through `method_params.solver`.
- `model_based_only`, `standalone_rl`, and the MGA ablation configurations are
  comparison settings. They are not additional named paper algorithms.
- CEM and CMA-ES implementations exist in the source tree, but this V1
  catalogue lists algorithms with published task configurations. Source
  availability alone does not establish a supported task recipe.

## Integration details

MDOC's original paper and project include multi-robot coordination through
Conflict-Based Search. The V1 recipes in this repository expose its
single-robot planning component; use the linked original project for the
published multi-robot application.

The contact-control DIAL-MPC and PegasusFlow baselines retain their sampling
mechanisms under the shared task and rollout interface. ATACOM trains and
executes a policy in tangent-action coordinates. ISSA filters nominal policy
actions through simulator queries. These are task integrations, so comparisons
should report their configuration, learned assets, and total rollout cost.

SafeDiffuser uses a local CBF safety correction inside the vendored learned
diffusion architecture. DPCC likewise uses its dedicated learned-model
integration. Both require their optional Torch dependencies and trained assets.

MGA stands for **Manifold Generative Annealing**. Its manuscript is titled
*Manifold Generative Annealing for Contact Constraint-aware Impedance Control
with Reinforcement Learning Prior*. A public paper URL has not been verified,
so this catalogue links the implementation. V1 qualifies its CPU path; GPU
qualification and humanoid push/walking remain deferred.
