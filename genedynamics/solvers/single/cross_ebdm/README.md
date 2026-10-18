# Cross-EBDM

This is the package for the proposed **Cross-EBDM algorithm**. It is a
single-solver implementation, alongside `acp`, `fada`, `dial`, and the other
method packages. It is not a repository-wide abstraction.

The package owns only:

- the `CrossEBDMSolver` frontend;
- Cross-EBDM planning and inference backends;
- Cross-EBDM-specific learned response and transport models.

Shared solver machinery used by more than one algorithm belongs in
`genedynamics.solvers.common`. Public learning infrastructure belongs in
`genedynamics.learning`. Transfer task controllers and physics queries belong
in `genedynamics.tasks.transfer` and
`genedynamics.envs.domains.manipulation.transfer_workcell`, respectively.

