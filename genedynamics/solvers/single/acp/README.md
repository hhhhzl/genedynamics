# ACP solver

ACP follows the repository's single-solver package contract:

- `acp.py` is the stable frontend.
- `backend_impl.py` defines the backend surface.
- `backends/acp_torch.py` performs policy inference.
- `backends/compliance.py` and `backends/spatial.py` preserve ACP command math.
- `backends/native_source.py` verifies and constructs the pinned upstream model.

Learned model topology and training live in `genedynamics.solvers.single.acp.learning`. The
four-task observation/action binding lives in
`genedynamics.experiments.plugins.methods.transfer.acp`. Demonstrators live in
`genedynamics.tasks.transfer.demonstrators`. The solver therefore does not own
task state, experiment result paths, training loops, or paper workflows.

The frontend accepts native sparse observation histories and the current tool
pose, then returns reference poses, virtual-target poses, stiffness matrices,
scalar stiffness, and waypoint times. Native 19D/38D action semantics and strict
checkpoint loading are preserved.

The port is derived from `yifan-hou/adaptive_compliance_policy` at commit
`74d68eca3bfb5cb59da131c6351d9e1c9b129ff6`. See `SOURCE.json`, `LICENSE`, and
`LICENSE.multimodal`; the checkout pin is in
`requirements/baselines/sources.lock.json`.
