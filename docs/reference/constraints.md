# Constraint solvers

Compose [CBF action filters](https://github.com/hhhhzl/genedynamics/tree/release/v1-open-source/genedynamics/core/constraints/action_filters),
[CFS trajectory projections](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/action_filters/cfs_qp_full.py)
and [adaptive augmented-Lagrangian scheduling](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/schedulers/ConstraintScheduler/almadaptive/alm_adaptive.py)
with the numerical interfaces below.

| Solver | What it provides | Implementation | Upstream reference |
| --- | --- | --- | --- |
| **Closed form** | Euclidean halfspace projection, box clipping and unconstrained solves for special cases. | [ClosedFormSolver](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/solvers/closed_form.py) | Local implementation |
| **JAXopt OSQP** | A general convex-QP interface built on `jaxopt.OSQP`. | [JAXOPTOsqpSolver](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/solvers/jaxopt_osqp_solver.py) | [JAXopt documentation](https://jaxopt.github.io/stable/quadratic_programming.html) |
| **OSQP** | Sparse CPU convex-QP solves through the OSQP Python interface. | [OSQPSolver](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/solvers/osqp_solver.py) | [OSQP documentation](https://osqp.org/docs/index.html) |
| **CVXOPT** | CPU convex-QP solves, including trajectory smoothness and slack formulations. | [CVXOPTSolver](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/solvers/cvxopt_solver.py) | [CVXOPT documentation](https://cvxopt.org/userguide/coneprog.html#quadratic-programming) |

Install optional numerical packages with:

```bash
python -m pip install -e ".[optimization]"
```

## Compose constraint handling

| Component | Use it for | Implementation |
| --- | --- | --- |
| **CBF filters** | Correcting actions along rollouts, including Cartesian-to-joint Jacobian lifting. | [Closed-form](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/action_filters/cbf_closed_form.py) · [QP filter](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/action_filters/cbf_qp.py) · [Joint lift](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/action_filters/cbf_qp_joint_lift.py) |
| **CFS projections** | Building local obstacle halfspaces and correcting per-step or full-horizon actions. | [Per step](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/action_filters/cfs_qp_perstep.py) · [Full horizon](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/action_filters/cfs_qp_full.py) · [Joint lift](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/convexify/cfs/action_joint_lift.py) |
| **Adaptive ALM scheduling** | Updating feasibility penalties and projection effort from constraint feedback. | [ALMAdaptiveConstraintScheduler](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/schedulers/ConstraintScheduler/almadaptive/alm_adaptive.py) |

CBF and CFS identify constraint-handling methods; the numerical solver is a
separate choice. See the original [CBF-QP paper](https://arxiv.org/abs/1609.06408)
and [CFS paper](https://arxiv.org/abs/1709.00627) for the underlying formulations,
then inspect [MDOC and MD-COAS recipes](../recipes/index.md) for their integration.

## Choose the right solve path

- **Closed form is specialized.** Its halfspace and box paths perform Euclidean
  projection and clipping. They are not general QP solves for arbitrary Hessians.
- **Fast JAX helpers are approximate projections.** Functions such as
  `solve_slack_qp_jax()` use finite, projection-style updates. They are distinct
  from the `JAXOPTOsqpSolver.solve_qp()` interface that invokes `jaxopt.OSQP`.
- **QPAX is an optional operator fallback.** The
  [per-step operator](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/core/constraints/operators/qp/per_step_filter.py)
  can call QPAX; this repository does not provide a separate QPAX solver class.

Use the [compatibility matrix](compatibility.md) for backend and device status.
Constraint margins, slack, iteration limits and the selected task determine the
behavior of an integration; a solver name alone does not establish task safety
or numerical convergence.
