# Predictive execution and control

The RecoveryMimic integration branch adds reusable contracts for interactive execution, episode
data and differentiable control. These mechanisms do not import a particular robot or tracker.

## Optional runtimes

Install `.[learning-torch]` for Torch predictive control or `.[jax]` for the JAX numerical runtime.
JAX is now an optional dependency of the base package; existing `optimization`, `simulation` and
`all` extras retain the V1 JAX requirement. Public exports load their implementation on first use.
Selecting a backend that was not installed raises an import/dependency error, without silently
switching its mathematical behavior.

`core.backends.differentiation.make_runtime` provides instance-local NumPy, Torch and JAX
implementations. NumPy derivatives use finite differences and are a diagnostic reference, not an
autodiff implementation. Physics engine and tensor runtime are separate selections.

## Interactive execution

`core.backends.execution` defines a capability record, registry and `ExecutionStep`. The latter
contains both the true pre-reset transition and the next collector carry. A simulator may expose
an exact, approximate or unsupported snapshot capability; callers must check before branching.
These interfaces coexist with legacy task and physics adapters.

## Dynamics and data

`learning.dynamics.composition.ClosedLoopRollout` combines a controller, reference generator,
physical predictive model, observation/history updates and task progression. It retains input
derivatives through a frozen controller. The application supplies all state and task semantics.

`learning.datasets.sequences` stores transition-aligned episodes with checksums and original
sequence/split identity. Sampling stays within an episode and includes padding/burn-in masks.
Persisted shards cross the host/device boundary explicitly; model rollout does not.

## Optimization

`solvers.single.budgeted_repair` accepts objective and normalized inequality callbacks, where
positive residual means violation. It computes a constraint-pullback metric, applies local
halfspace corrections within a trust radius, and re-evaluates nonlinear candidates. Its result
reports predicted feasibility, actual elapsed time and deadline termination. It is not a physical
safety certificate; a single derivative call cannot be preempted by its wall-clock budget.

`solvers.single.model_mppi` consumes the same problem and budget for a matched-model sampling
comparison. Both preserve native tensors inside the predictive callbacks.

## Acceptance

```sh
python -m pytest test/unit/predictive_control test/unit/sequence_data
```

These tests cover numerical repair across three runtimes, derivatives through frozen controllers,
episode boundaries/checksums/splits, optional imports and matched-model sampling. They do not
qualify a particular GPU, physics engine, learned model or robot. Downstream execution acceptance
remains the application's responsibility.

## Learning and provenance

`learning.networks.torch`, `learning.dynamics.training`, `learning.policy_optimization.torch`
and `learning.checkpoints.torch` are explicit optional Torch modules. The multistep trainer
consumes application-supplied dynamics/error callbacks and valid masks. PPO advantage recursion
stops at resets and separately handles terminal versus timeout bootstrapping. Checkpoint manifests
verify bytes, metadata and restoration level; a learner checkpoint is not a simulator snapshot.

`solvers.single.reference_mga` adapts prior-centered annealing, augmented feasibility and
constraint-pullback noise to the same tensor `ControlProblem`. It does not replace the original
Brax/spline/impedance MGA implementation. Applications must describe this adaptation in comparisons.

Episode append locks serialize split checks across Linux collector processes. Artifact utilities
record actual checkout revisions, source-tree digests, checksums and resolved configuration. Solver
`model_evaluations` counts explicit candidate assessments; instrument predictive callbacks for total
model-forward/plant-step counts, and measure derivative costs by actual elapsed time.
