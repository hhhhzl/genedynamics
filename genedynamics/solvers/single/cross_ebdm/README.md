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

## Transport and execution semantics

The learned residual is added to the target task controller's `base_proto`.
Source commands and source history are conditioning inputs; the residual is not
generally added to the raw source action. The bounded output is a candidate
request. Public feedback decoding, task-owned channel projections, recovery
rules, and any explicitly bound runtime selector determine the executed request.

Three decisions have distinct meanings:

- The detached training gate enables source-response anchoring in the offline
  loss. The task term, shared risk penalty, and command-change penalty remain
  active according to their training contract.
- Offline checkpoint selection compares a candidate with its matched task-only
  reference on a declared validation proxy. A conditional two-checkpoint bound
  concerns that same short-horizon proxy; it does not certify episode success
  or physical safety.
- Task-owned runtime admission may compare candidate and reference requests
  using the frozen response ensemble. A controller with this binding performs
  online model calls. Its results and latency do not describe a network-only
  deployment path.

`learning/residual_transport.py`, `learning/task_transport.py`, and
`learning/gated_transport.py` own the corresponding model and loss definitions.
Runtime bindings and selectors remain in
`genedynamics.tasks.transfer.controllers`. A backbone comparison must keep these
execution components fixed and use matched training starts, continuation
budgets, and checkpoint-selection rules for both objectives.

## Primary paper ablations

`learning/ablation.py` defines the three matched treatments used by the paper:
`no_calibration`, `no_response_anchoring`, and
`unconditional_response_anchoring`.  They respectively change only the response
calibration role, the response-loss weight, or the anchoring gate.  All retain
the task band.  The shared lock is recorded in
`configs/arm/keyed_insertion_recovery/ablation/clean_treatment_contract.json`.
Legacy `band_only`, `ungated_full`, `frozen_warm`, and `matching_no_band`
artifacts are not reusable evidence for this matched comparison.

