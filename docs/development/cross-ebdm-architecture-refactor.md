# Cross-EBDM architecture refactor

This refactor replaces phase-specific experiment workflows with the repository's
normal task, environment, solver, learning, config, and paper-entry boundaries.
Actor equivalence and the GPU0-only confirmation execution passed on 2026-10-18.
The subsequent CPU audit found an omitted physical-mode predicate, incomplete
worst-group aggregation, and a missing folded/noisy T4 reset binding. Execution
integrity is retained; the old run does not certify the full task contract or
the declared T4 initial-state distribution. Mode evaluation and durable evidence
recording have now been repaired for development runs; the next formal version
still requires qualification and a fresh independent cohort. The decision record is
`cross-embodiment-current-results-and-final-experiment-plan.md`.

## Recovery point

The complete pre-refactor worktree is stored at
`results/_architecture/cross_ebdm_pre_refactor_20261017`. Existing experiment
results were not rewritten. The snapshot is the source for recovering an old
script or checking a historical config after the active tree is cleaned.

## 1. Task and environment ownership

- `genedynamics/tasks/transfer` owns the four task specifications, episode
  records, observations, evaluation rules, protocols, routes, and task-level
  feedback laws.
- `genedynamics/envs/domains/manipulation/transfer_workcell` owns workcell
  construction, numerical providers, simulation state, reset, and stepping.
- `genedynamics/envs/adapters/transfer_task.py` is the public CPU task adapter.
- `genedynamics/envs/adapters/acp_transfer.py` adapts ACP commands to that
  environment without placing physics in a solver package.
- The opt-in `compliant_keyed_insertion_recovery` extension follows the same
  boundaries: task contract/frame/evaluation in `tasks/transfer`, passive socket
  physics in the existing workcell, and a task adapter exposing robot encoders
  separately from measured fixture pose/twist. Its configs and results use the
  new physical task name; the original static T3 contract stays unchanged.
  `tasks/transfer/compliant_socket_qualification.py` owns ringdown validation,
  episode collection, evidence, and report generation; its paper command only
  parses a recipe and invokes that work unit.

## 2. Learning and solver ownership

- `genedynamics/learning` contains only solver-independent priors, datasets,
  checkpoint utilities, reliability utilities, and general training entrypoints.
- `genedynamics/solvers/single/cross_ebdm` is the independent Cross-EBDM method
  package. It owns planning, numerical inference, and its method-specific
  `learning` implementation.
- `CrossEBDMSolver` is the method entrypoint. The environment entrypoint is
  separately named `TransferWorkcell`; the old generic `CrossEmbodiment` aliases
  have been removed.
- A utility moves to `genedynamics/solvers/common` only after more than one
  solver owns the same contract. Cross-EBDM-only response and transport code
  remains inside the Cross-EBDM method package.
- Every comparison method has one package under `genedynamics/solvers/single/<method>`:
  ACP, FADA, Human2Sim2Robot, Play2Perfect, DIAL, and response MPC.
- ACP follows the standard frontend/backend layout. Its learned models live in
  `genedynamics/solvers/single/acp/learning/models` and its demonstrators are task owned.
- Task controllers and method/task bindings live under
  `genedynamics/tasks/transfer/controllers`; experiment plugins only register
  and assemble them.
- Runtime packages do not import `scripts`.
- Fixture-aware public trajectory serialization remains in the shared dataset
  package; Cross-EBDM's segmented response features/labels remain in its own
  `learning/response.py`. The paper data-preparation command only parses a
  task-local recipe and invokes that implementation.
- Shared datasets and workcells do not import method packages, and one solver
  cannot import another solver's private learning implementation.

## 3. Unified experiment path

The four transfer workcells are registered environment plugins. The shared task
reference is executable, and `transfer_task` supplies evaluator-owned common
outcome metrics. The method plugin is a thin loop around actors created by
`genedynamics.tasks.transfer.controllers.bindings`; solver packages continue to
own algorithms and checkpoint formats. Cross-EBDM, task-only, FADA, ACP,
Human2Sim2Robot, DIAL, response MPC, public geometry, and public source prior
have named bindings. Learned methods reject execution until their SHA-bound
package artifacts are supplied.

`genedynamics.experiments.runner` no longer contains Cross-EBDM workflow
branches. A two-control-step obstacle-transfer run completed through the normal
runner on 2026-10-17 and wrote its isolated evidence under `/tmp`.

## 4. Config and result layout

Each physical task owns `_base.yaml` plus `main`, `baseline`, `ablation`, and
`validation` directories. The canonical evidence layout follows existing
repository conventions:

`results/<embodiment>/<task>/<category>/<method>/level_<condition>/seed_<seed>`

There is no paper-project or run-id directory layer. Active configs use
functional method/ablation names. Their protocol status is
`development_revision_pending`; the accessed confirmation cohort is explicitly
closed. The original base configs and source are retained in the completed
run's `source_snapshot.zip`. New CPU diagnostics use explicit task-owned
`validation` configs and isolated results. A `run_class` label or output-root
change alone cannot make the old cohort admissible as fresh confirmation.

## 5. Paper entrypoints

The required boundary for `scripts/paper/cross_ebdm` is thin commands for the
main matrix, baselines, ablations, result verification/summary, and LaTeX build.
Physics, training, evaluation, and report generation belong to their runtime
owners. The new fixture qualification, response-data preparation, and nominal
learning diagnostic commands follow that boundary. Existing T3/T4 completion
training scripts still contain substantial training logic; the present change
only rejects incompatible fixture records at their inputs. That remaining
migration debt is not evidence that the entire paper directory is clean, and
those scripts must not be copied as the pattern for the new backbone work.
The LaTeX tree remains the manuscript draft and does not store experiment reports.

The existing `summarize_results.py` also invokes CPU-only frozen-outcome
analysis in `genedynamics/experiments/utils/metrics.py`. It writes derived
statistics beside the existing protocol evidence; SHA-bound execution files
remain immutable. Statistical code stays out of solver and learning packages.

`verify_results.py --evidence ...` invokes task-owned CPU reevaluation of
compressed episode records. Generic artifact compression, hashes, trajectory
serialization, and retained-failure summaries stay in the experiment framework;
the framework contains no task-specific evaluation or method dispatch.

## 6. Release gate before the next experiment stage

The refactor is admitted only after all of the following pass:

1. architecture tests and package import tests;
2. dry-run validation of every active YAML;
3. one real smoke episode per environment through the unified runner;
4. behavior comparison against retained pre-refactor evidence for each migrated actor;
5. package-owned checkpoint bindings for every main-table method;
6. no second root-level `solvers/` tree and no task actors or demonstrators in
   baseline solver packages.

The superseded runner/config trees, duplicate root solver tree, and the
SHA-reviewed stale tests and development notes have been removed. Their exact
deletion receipt is retained under `results/_architecture`. Actor bindings and
equivalence were subsequently admitted in
`results/_architecture/cross_ebdm_architecture_equivalence_20261018/report.json`
and the GPU0-only confirmation completed. The current release issue is the
evaluation/evidence contract documented above, rather than pending migration.
