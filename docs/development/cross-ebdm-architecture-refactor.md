# Cross-EBDM architecture refactor

This refactor replaces phase-specific experiment workflows with the repository's
normal task, environment, solver, learning, config, and paper-entry boundaries.
The next experiment stage remains closed until the new actor bindings reproduce the retained results.

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
- Shared datasets and workcells do not import method packages, and one solver
  cannot import another solver's private learning implementation.

## 3. Unified experiment path

The four transfer workcells are registered environment plugins. The shared task
reference is an executable method plugin, and `transfer_task` supplies common
outcome metrics. Cross-EBDM, FADA, ACP, Human2Sim2Robot, Play2Perfect, DIAL, and response MPC
have named plugin slots that reject execution until a frozen package-owned actor
binding is supplied. This is deliberate: old script workflows cannot silently
write new paper evidence.

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
`architecture_refactor_pending`, so canonical formal execution is blocked while
actor bindings and equivalence checks are unfinished.

## 5. Paper entrypoints

`scripts/paper/cross_ebdm` contains only thin commands for the main matrix,
baselines, ablations, result verification/summary, and LaTeX build. Physics,
training, evaluation, and report generation do not live in the paper directory.
The LaTeX tree remains the manuscript draft and does not store experiment
reports.

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
deletion receipt is retained under `results/_architecture`. Until items 3–5 are
complete, the formal protocol stays blocked.
