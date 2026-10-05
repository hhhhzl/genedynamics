# MGA Paper Experiment Integration and Run Plan

> Status: implementation plan, not an experiment result.
>
> Audit baseline: repository `5e10ca8` on 2026-08-24. The worktree was dirty at
> audit time, and several PegInsert/H1 files were still untracked. The scoped
> implementation snapshot containing this record tracks those files. Every
> formal run must use a clean checkout of that commit and record it in the
> standard experiment output.

## 1. Purpose

This document defines the complete implementation and execution order for the
three MGA paper experiments:

1. Franka/Panda surface scanning and polishing on rigid, compliant, unseen, and
   spatially varying surfaces.
2. Franka/Panda peg insertion under nominal, pose-OOD, and sensing-OOD contact.
3. Unitree H1 push-to-line, including force-step validation, fixed-stance
   pushing, unjamming/yaw correction, and walk-and-push.

The plan has four non-negotiable goals:

- Formal experiments enter through
  `python -m genedynamics.experiments.runner <config.yaml>`.
- Solvers remain algorithms under `genedynamics/solvers`; experiment
  orchestration, configuration expansion, metrics, artifacts, and reports do
  not live inside an individual solver package.
- Formal outputs follow the repository's established paper layout:
  `level_<suite>/seed_<seed>/results.json`, per-level `summary.json`, and
  `overall_summary.json`, with trajectory images/GIFs next to each seed.
- Every paper claim must be no stronger than the evidence produced by the
  locked formal matrix. Development results may motivate a hypothesis, but may
  not be relabeled as formal evidence.

The former solver-local development harness has been retired. Contact
experiments now enter exclusively through `genedynamics.experiments.runner`;
controller composition lives in the existing method-plugin layer.

---

## 2. Immutable architectural rules

### 2.1 Ownership

The final ownership boundary is:

| Concern | Owner |
|---|---|
| MGA, DIAL, MPPI, PegasusFlow, ISSA, ATACOM solver logic | `genedynamics/solvers/` |
| Robot models and task dynamics | `genedynamics/envs/` |
| Algorithm adapters used by the unified runner | `genedynamics/experiments/plugins/methods/` |
| Task adapters used by the unified runner | `genedynamics/experiments/plugins/environments/` |
| Metric extraction and shared metric math | `genedynamics/experiments/plugins/metrics/` and `genedynamics/evaluation/` |
| Per-seed visualization plugins | `genedynamics/experiments/plugins/visualizations/` |
| Cross-run aggregation, verification, and result rendering utilities | `genedynamics/experiments/utils/` |
| Paper entry scripts | `scripts/paper/mga/` |

`scripts/paper/mga` contains orchestration only. It must not implement a
solver, task reward, metric, statistical test, renderer, or result parser.
Runtime metric adapters remain under `genedynamics/experiments/plugins/metrics/`;
cross-run result/statistics utilities remain under
`genedynamics/experiments/utils/`. Task-script directories are not an alternate
home for either layer.

### 2.2 Algorithms are not MGA variants

The formal algorithms are independent algorithm-level entries:

- DIAL
- MPPI
- PegasusFlow
- ISSA
- ATACOM
- standalone RL
- Model-based Only
- MGA

Each algorithm has its own YAML and its own method plugin name. PegasusFlow,
ISSA, ATACOM, MPPI, and DIAL must not be encoded as cosmetic variants of the
MGA YAML. Shared adapter code is allowed internally, but the runner sees
independent algorithms.

### 2.3 Device is not a scientific condition

Canonical config directories and experiment names must not contain `cpu`,
`mac`, `gpu`, `debug`, `v3`, or `v4`. CPU/GPU selection is a runtime deployment
choice. The scientific protocol fixes task physics, seeds, horizons, sampling
budget, checkpoints, and metrics.

### 2.4 Preserve legacy evidence

The following current result roots are development/legacy evidence and must not
be overwritten, moved, or deleted during integration:

```text
results/arm/impedence/
results/arm/impedence/cpu_controllability_geometry/
results/arm/peg_insert_cpu/
results/humanoid/box_push/
```

New formal runs use new standard-runner result roots. Old results are used for
equivalence checks and for deciding which claims need further testing.

---

## 3. Current evidence and permitted claims

This section freezes what the current files actually demonstrate. It prevents
the paper narrative from drifting ahead of the data.

### 3.1 Surface scan evidence

The surface implementation supports the following families.

Rigid geometry:

```text
plane, cylinder, convex, bumpy, unseen
```

Soft geometry:

```text
plane, cylinder, convex, bumpy, unseen
```

Hybrid spatial stiffness maps:

```text
stripes, center_hard, center_soft
```

The latest eight-algorithm comparison contains only two seeds and only these
three suites:

```text
rigid_cylinder, soft_cylinder, soft_unseen
```

Representative means from that comparison are:

| Suite | Method | Surface tracking | Common tracked force error | Force violation | Deformation CVaR95 |
|---|---:|---:|---:|---:|---:|
| rigid cylinder | MGA | 1.610 mm | 3.020 N | 0.0 | 0.370 mm |
| rigid cylinder | DIAL | 4.468 mm | 5.413 N | 0.0 | 0.502 mm |
| soft cylinder | MGA | 2.267 mm | 4.930 N | 0.0 | 0.471 mm |
| soft cylinder | DIAL | 4.582 mm | 6.307 N | 0.0 | 0.534 mm |
| soft unseen | MGA | 1.356 mm | 0.914 N | 0.0 | 0.413 mm |
| soft unseen | DIAL | 8.745 mm | 6.192 N | 0.0 | 1.031 mm |

These results support the following restricted claim:

> Realization-aware manifold/model-based control improves the
> precision--force--deformation tradeoff relative to unconstrained or weakly
> constrained diffusion/sampling baselines under the tested rigid, compliant,
> and unseen conditions.

They do not support either of the following claims:

- MGA wins every scalar metric against every baseline.
- The RL prior is responsible for the scanning improvement.

Model-based Only and ATACOM are slightly better than MGA on some scalar
metrics. For example, on soft unseen, their mean tracking errors are about
1.317 mm and 1.235 mm versus MGA's 1.356 mm. The formal paper must report
the complete Pareto picture instead of hiding this.

Older development results cover rigid/soft plane, cylinder, bumpy, and unseen,
usually with only one seed and a previous method definition. `convex` has no
completed result in the audited roots, and the hybrid maps do not have a formal
matrix. Those files are exploratory evidence only.

### 3.2 PegInsert evidence

The current canonical-v4 result root contains four algorithms, three suites,
and ten seeds per suite. Define strict safe insertion success as:

```text
insertion_success
AND NOT any_force_torque_violation
AND NOT jammed_once
```

The existing results give:

| Suite | MGA | DIAL | Model-based Only | ATACOM |
|---|---:|---:|---:|---:|
| ID | 80% | 40% | 0% | 0% |
| Pose OOD | 60% | 10% | 0% | 0% |
| Sensing OOD | 100% | 40% | 0% | 0% |

MGA has zero force/torque violations and zero jams in these 30 episodes.
The baselines often reach 100% raw insertion success but do so with violations
or jamming. The supported claim is therefore:

> Reliability-aware model-based revalidation converts raw insertion progress
> into a substantially better strict safe-success tradeoff under contact-mode
> switching and OOD pose/sensing conditions.

The current files do not isolate the causal contribution of the learned RL
prior or of learned reliability. The following conclusions remain unproven:

- RL proposals are better than Gaussian proposals under an otherwise identical
  safety shell.
- Learned reliability is better than a fixed/heuristic gate.

Those claims require the ablations specified later in this plan.

### 3.3 H1 push evidence

The audited H1 results compare DIAL with `model_based_only`; they are not a
comparison with the final RL-prior/reliability MGA.

- Unjamming unjamming: model-based MGA succeeds safely on 2/2 seeds; DIAL succeeds on
  0/2.
- Walk-and-Push walk-and-push: both methods succeed safely on 2/2 seeds.
- Force Regulation and Fixed-Stance Push currently have only one completed seed per method.

The supported claim is:

> Task-owned contact geometry enables rear-contact unjamming/yaw correction,
> and the same model-based contact-control interface is executable in a
> coupled whole-body walk-and-push task.

The current evidence does not support:

- a claim that MGA has been validated on H1;
- a claim that MGA is statistically superior on Walk-and-Push;
- a claim that RL prior or learned reliability improves whole-body pushing.

---

## 4. Paper research questions fixed to the evidence

### RQ1: Geometry and compliance in surface scanning

> Does realization-aware manifold optimization preserve path accuracy and
> force/deformation safety across geometry and compliance shifts?

Primary mechanisms:

- cumulative path/contact manifold;
- short-horizon controllability/realization lift;
- model-based rollout and refinement;
- log-SPD/adaptive impedance;
- stiffness adaptation on hybrid spatial maps.

Primary comparisons:

- MGA versus DIAL/MPPI/PegasusFlow/ISSA;
- MGA versus Model-based Only and ATACOM to expose the actual Pareto
  frontier;
- geometry and stiffness ablations on the hybrid/complex surfaces.

This experiment is not the primary proof of the RL prior.

### RQ2: Safe contact switching in PegInsert

> Can reliability-aware model-based revalidation achieve high strict safe
> insertion success under contact switching and OOD pose/sensing errors?

Primary mechanisms:

- learned proposal prior;
- short-horizon model-based refinement;
- learned reliability;
- incumbent/refined candidate revalidation;
- task-owned retract/zero-force emergency candidate.

Primary headline metric:

```text
strict_safe_insertion_success
```

Raw success remains visible as a secondary metric. The paper must explicitly
show when a baseline achieves raw success by exceeding force/torque limits or
jamming.

The RL-prior claim is allowed only if the no-prior and standalone-RL comparisons
show an improvement under the same safety shell.

### RQ3: Contact-geometry transfer to whole-body manipulation

> Does task-owned contact geometry enable unjamming and transfer from
> fixed-stance object interaction to coupled whole-body walk-and-push control?

Force Regulation is controller validation, Fixed-Stance Push is nominal/OOD fixed-stance pushing, Unjamming is the
main geometry task, and Walk-and-Push is a whole-body generalization extension. Walk-and-Push is not
presented as a win unless the formal multi-seed results actually show one.

---

## 5. Target unified-runner architecture

### 5.1 Generic suite support

`ExperimentConfig` currently supports `obstacle_levels`, which is insufficient
for contact suites with different physics and task parameters. Add an optional,
generic `suites` field. A suite is a named evaluation condition, not an MGA
special case.

Required suite schema:

```yaml
suites:
  - name: rigid_plane
    level: plane
    env_params:
      medium: rigid

  - name: soft_unseen
    level: unseen
    env_params:
      medium: soft
      s4_stiffness_range: [800.0, 2000.0]
      s4_friction_range: [0.05, 0.2]

  - name: pose_ood
    level: wide
    execution_env_params:
      level: ood
      ood_position_range: 0.001
      ood_angle_range: 0.0122173
```

Compatibility contract:

- Configs without `suites` run exactly as before using `obstacle_levels`.
- Configs with `suites` run `suite x seed`.
- Per-suite `env_params`, `execution_env_params`, and narrowly justified
  `method_params` overrides are deep-merged over the common config.
- Sampling-budget keys may not differ between algorithms in the same suite.
- The CLI adds `--suite <name>` while retaining the existing `--level` and
  `--seed` behavior.
- The output label is `level_<suite-name>` so the existing level/seed result
  hierarchy and downstream report discovery remain valid.

### 5.2 Generic base inheritance

Move the useful `base:` deep-merge behavior from the MGA-private config loader
into `ExperimentConfig.from_yaml`. Existing self-contained configs continue to
load unchanged. Canonical algorithm YAMLs inherit a shared task protocol,
preventing surface lists, seed lists, and budgets from drifting between
algorithms.

Base resolution requirements:

- paths are relative to the referring YAML;
- cycles are detected and rejected;
- dictionaries deep-merge;
- lists replace rather than append;
- the resolved config is stored in `results.json` metadata;
- `base` is not passed into the dataclass constructor.

### 5.3 Environment plugins

Add unified-runner adapters for:

```text
manipulator_surface_scan
manipulator_peg_insert
humanoid_box_push
```

The adapters call the existing environment factories. They do not reimplement
robots, task rewards, contact geometry, or impedance control.

The environment-plugin contract must gain an optional task-owned reset hook.
Contact tasks need the actual structured Brax/MJX reset state; the generic 2-D
start-position sampler is not valid for them.

The model environment and hidden/OOD execution environment remain distinct.
The adapter exposes both to the method plugin, and metrics always evaluate the
actually executed environment/state sequence.

### 5.4 Method plugins

Register independent algorithm plugins for the eight algorithms. Internally,
the receding-contact plugins may share a small adapter that:

1. receives the already-created task environment(s);
2. calls the existing solver/controller factory;
3. runs `run_receding` for the configured number of control steps;
4. returns the actually executed states, actions, per-step infos, initial state,
   and timing in the unified result schema.

The adapter must not replay an action sequence to fabricate the executed
trajectory. PegInsert and H1 can bifurcate under replay. Formal metrics must use
the states collected during the real receding-horizon execution.

MGA has one public algorithm name, but the resolved component contract is
recorded per task. The current internal methods are inconsistent across tasks
(`mga_controllable`, `mga_controllable_gate`, and `mga`); this must be made
explicit and validated before any H1 result is labeled MGA.

### 5.5 Direct trajectory and metric context

The unified runner must accept a method plugin returning a ready `Trajectory`
without coercing structured MJX states to NumPy vectors. It must also pass the
following metric context:

- actual `x0`;
- evaluation seed;
- actual execution environment;
- planning/execution wall time;
- method diagnostics and per-step infos.

This is a generic runner capability. There must be no branch such as
`if method == "mga"` in metric computation.

### 5.6 No-obstacle contact task

Contact tasks do not use the runner's default `box2d` obstacle generator. Add a
generic empty/no-obstacle generator or make the obstacle stage optional through
the plugin contract. Do not create dummy boxes to satisfy the old runner path.

### 5.7 Metrics registration and aggregation

Register all three existing general metric plugins in `runner.py`, including
the currently omitted PegInsert plugin.

The summary writer must aggregate nested scalar dictionaries returned by
`GeneralMetricsPlugin`. Per-seed `results.json` is authoritative; per-level and
overall summaries store count, mean, standard deviation, and the configured
confidence interval/statistical fields.

NaN/Inf values must be converted to `null` or omitted with an explicit missing
count so the output is strict JSON and paper aggregation cannot silently treat
missing diagnostics as zero.

---

## 6. Canonical configuration layout

```text
configs/
├── arm/
│   ├── surface_scan/
│   │   ├── _base.yaml
│   │   ├── main/mga.yaml
│   │   ├── baseline/
│   │   │   ├── dial.yaml
│   │   │   ├── mppi.yaml
│   │   │   ├── pegasusflow.yaml
│   │   │   ├── issa.yaml
│   │   │   ├── atacom.yaml
│   │   │   ├── standalone_rl.yaml
│   │   │   └── model_based_only.yaml
│   │   └── ablation/
│   │       ├── no_controllability_geometry.yaml
│   │       ├── no_retraction.yaml
│   │       ├── no_rl_prior.yaml
│   │       └── no_learned_reliability.yaml
│   └── peg_insert/
│       ├── _base.yaml
│       ├── main/mga.yaml
│       ├── baseline/<one-yaml-per-algorithm>
│       └── ablation/
│           ├── no_rl_prior.yaml
│           └── no_learned_reliability.yaml
└── humanoid/
    └── push_to_line/
        ├── _base.yaml
        ├── main/mga.yaml
        ├── baseline/<one-yaml-per-algorithm>
        └── ablation/
            ├── no_rl_prior.yaml
            ├── no_learned_reliability.yaml
            ├── no_controllability_geometry.yaml
            ├── no_retraction.yaml
            └── no_stiffness.yaml
```

No algorithm is represented by several tuning YAMLs in the formal tree. Search
and diagnostic configs stay outside this tree. Once a formal YAML is locked,
runtime semantic overrides of physics, horizon, seed set, or sampling budget
are forbidden.

### 6.1 Surface-scan suite list

The main geometry/compliance list is:

```text
rigid_plane
rigid_cylinder
rigid_convex
rigid_bumpy
rigid_unseen
soft_plane
soft_cylinder
soft_convex
soft_bumpy
soft_unseen
```

The hybrid mechanism list is:

```text
hybrid_stripes
hybrid_center_hard
hybrid_center_soft
```

All eight algorithms and all four causal ablations run all thirteen suites.
This evaluates learned components, controllability geometry, and retraction
across rigid, soft, unseen, and hybrid conditions without selecting only the
hybrid cases where contact/material switching is most pronounced.  The main
text may emphasize the most diagnostic hybrid comparisons, but the complete
paired matrix remains available in the appendix.

### 6.2 PegInsert suite list

```text
id_wide
ood_pose
ood_sensing
```

Delay, tight-clearance, stiffness, friction, and chamfer extremes remain stress
tests until they have a separately locked success contract. They must not be
quietly merged into a formal suite after inspecting formal seeds.

### 6.3 H1 suite list

```text
force_regulation_15n
force_regulation_30n
fixed_stance_push_nominal
fixed_stance_push_ood
unjamming
walk_and_push
```

Force Regulation has a force-step success contract, not a line-reaching success contract. Its
metrics are rise time, settling time, overshoot, steady-state force error, force
violation, and balance margin. Fixed-Stance Push--Walk-and-Push use task-specific safe push success.

All eight algorithms run all six H1 suites.  The four primary H1 ablations are
now aligned with Surface and run every suite, so a row has the same causal
meaning across the two tasks:

```text
no_rl_prior:                    all suites of the four Humanoid tasks
no_learned_reliability:         all suites of the four Humanoid tasks
no_controllability_geometry:    all suites of the four Humanoid tasks
no_retraction:                  all suites of the four Humanoid tasks
no_stiffness:                   force_regulation_15n, force_regulation_30n
```

The first four are the cross-task causal matrix.  The final one is retained as
the Force Regulation-specific impedance diagnostic.  The former Unjamming `no_tangent` formal YAML
is removed rather than renamed: tangent projection and the finite-difference
realized-response lift are different mechanisms, so relabeling its old results
would not be a valid controllability-geometry ablation.

---

## 7. Locked budgets and fairness rules

The currently validated CPU evaluation budgets are the starting point for the
formal protocol:

| Task | Closed-loop steps | Nsample | Hsample | Hnode | Ndiffuse |
|---|---:|---:|---:|---:|---:|
| Surface scan | 100 | 64 | 16 | 4 | 2 |
| PegInsert | 64 | 64 | 8 | 4 | 2 |
| H1 push | 100 | 64 | 16 | 4 | 2 |

These values may be changed once, before the formal lock, through a documented
budget study. They may not differ between algorithms within a task. Reported
runtime uses the same device class and synchronization policy.

Evaluation seeds are paired across algorithms:

```text
10, 11, 12, 13, 14, 15, 16, 17, 18, 19
```

Rules for learned methods:

- MGA and standalone RL use the exact same RL checkpoint for a task.
- ATACOM uses its tangent-space checkpoint, trained with the same environment
  interaction budget and training-domain count as the corresponding learned
  methods.
- Policy-training seeds are distinct from evaluation seeds and recorded.
- The scanning ATACOM 200k versus Full/standalone-RL 2M mismatch must be removed
  before the paper comparison is valid.
- PegInsert currently uses a 200k learned-policy protocol. Retain it unless a
  preregistered budget study changes all learned methods together.
- H1 must use suite-aware frozen checkpoint bindings because its primitive
  dimension changes in Walk-and-Push and its ATACOM equality dimension changes in Unjamming:
  one raw PPO checkpoint shared by Full/standalone-RL/ISSA for the three fixed-base tasks, one raw
  PPO checkpoint for Walk-and-Push, and separate ATACOM tangent checkpoints for Force Regulation/Fixed-Stance Push,
  Unjamming, and Walk-and-Push.  A YAML path alone is not a valid binding; the resolved suite,
  action/observation dimensions, protocol, and checkpoint hash are audited.
- H1 requires a fixed-dimensional task-wide learned-reliability contract (or a
  preregistered fixed-stance/walk split if one contract is demonstrably
  impossible).  MGA is not run until the H1 policy, reliability, safety,
  and suite-aware checkpoint contracts pass development seeds.

Formal visualizations use a preregistered representative-seed rule, such as the
successful seed nearest the multi-metric median. A visually attractive or
"best" seed may be shown separately as a qualitative example but may not be
presented as representative.

---

## 8. Standard result contract

Formal output roots are:

```text
results/arm/surface_scan/
results/arm/peg_insert/
results/humanoid/push_to_line/
```

Each config sets an algorithm/group output directory. The unified runner then
creates the same leaf structure as previous paper experiments:

```text
results/<domain>/<task>/<group>/<algorithm>/
├── level_<suite>/
│   ├── seed_<seed>/
│   │   ├── results.json
│   │   ├── trajectory/
│   │   │   ├── trajectory.json
│   │   │   ├── trajectory_best.png
│   │   │   └── trajectory_best.gif
│   │   └── diagnostics/            # only declared task/method diagnostics
│   └── summary.json
└── overall_summary.json
```

`results.json` contains:

- suite name and seed;
- planning/execution timing;
- nested task metrics;
- method diagnostics;
- resolved component contract;
- resolved config snapshot;
- git commit and dirty flag;
- checkpoint identifiers/hashes for learned methods.

`trajectory/trajectory.json` contains serializable executed controls and the
task signals required to reproduce plots. Full structured MJX states need not
be serialized as enormous JSON; renderer-required state may be stored in the
existing compact binary artifact convention if necessary and referenced from
`results.json`.

Do not create a second MGA-only `metrics.json/manifest.json` formal schema.
Those files remain readable as legacy evidence but are not emitted by the new
formal path.

---

## 9. Required paper outputs

### 9.1 Surface scan

Per-suite data:

- safe scan success;
- maximum visited path coverage from the entire EE trajectory;
- realized progress;
- surface/tangential/normal tracking error;
- force MAE/CVaR/overshoot/violation;
- deformation peak/CVaR;
- contact loss/chatter;
- control and stiffness smoothness;
- runtime and sampling budget;
- gate/reliability correlations where defined.

Figures:

- safe success and coverage over all rigid/soft geometries;
- tracking--force--deformation Pareto plots;
- force/deformation/path time-series panels;
- hard versus soft/OOD generalization panel;
- hybrid stiffness-map adaptation and stiffness-command correlation;
- geometry/stiffness ablation plot.

### 9.2 PegInsert

Per-suite data:

- raw insertion success;
- strict safe insertion success;
- maximum/terminal insertion depth;
- completion time;
- peak/CVaR lateral and axial force;
- peak bending/torsional torque;
- force/torque violation rate;
- jam and recovery rate;
- cumulative safety cost;
- contact-mode transitions;
- emergency/revalidation rates;
- proposal-source weights and accepted improvement.

Figures:

- raw success versus strict safe success;
- success--peak-force Pareto;
- depth/force/torque/contact-mode timeline;
- safe recovery and rejected-jam qualitative examples;
- RL-prior/no-prior and learned-reliability ablations.

### 9.3 H1 push

Force Regulation data/figures:

- 15 N and 30 N force-step response;
- rise/settling time;
- peak/overshoot/steady-state MAE;
- force violation and balance margin.

Fixed-Stance Push--Walk-and-Push data/figures:

- safe push success;
- goal error/progress/completion time;
- yaw error and yaw recovery for Unjamming;
- fall and balance violation;
- force peak/CVaR/impulse;
- friction-cone violation/slip;
- non-hand collision;
- wall contact/force where wall contact is task-defined;
- Unjamming top-down box/yaw/contact trajectory;
- Walk-and-Push motion strip and executed GIF.

---

## 10. Paper-script contract

Create:

```text
scripts/paper/mga/
├── run_surface_scan.sh
├── run_peg_insert.sh
├── run_humanoid_push.sh
├── render_results.sh
├── summarize_results.sh
├── verify_results.sh
└── run_all.sh
```

Each run script only iterates over canonical YAMLs and invokes:

```bash
python -m genedynamics.experiments.runner <canonical-config.yaml>
```

Rendering, statistics, and cross-algorithm tables reuse the shared utilities in
`genedynamics/experiments/utils/{metrics,vis}.py` and are called from the paper scripts. The paper
scripts pass paths and safe filters; they do not pass semantic overrides such
as a different horizon, seed list, physics, reward scale, or sample budget.

`verify_results.sh` fails unless every expected algorithm/suite/seed has a
finite/explicitly-null metric record, matching resolved protocol, required
summary, and required visual artifact.

---

## 11. Complete implementation order

The phases below are sequential gates. A later phase does not start until the
previous phase's acceptance criteria pass.

### P0 — Freeze and inventory the current evidence

Actions:

1. Commit the scoped MGA, task, robot-abstraction, metric, config, and test
   changes. Do not include unrelated worktree changes.
2. Record the commit, Docker image identifier/digest, Python/JAX/Brax/MuJoCo
   versions, and current checkpoint hashes.
3. Inventory every current result by task, algorithm, suite, seed count,
   manifest commit, budget, and checkpoint.
4. Mark each result root as `legacy-development`, `validated-development`, or
   `formal-candidate`; do not rename the directories.
5. Freeze the current numerical reference records used by P5 equivalence tests.

Acceptance:

- Critical PegInsert/H1 source and configs are tracked.
- The worktree used for a formal run is clean.
- No formal claim depends on an untracked file or a dirty-only implementation.
- Legacy results remain byte-for-byte untouched.

Rollback:

- The pre-integration commit/tag is the rollback point.
- The unified framework, canonical configs, metrics, and tests are committed as
  one scoped implementation snapshot; generated result artifacts are excluded.

### P1 — Add backward-compatible config capabilities

Files primarily affected:

```text
genedynamics/experiments/framework/config.py
genedynamics/experiments/runner.py
test/integration/test_runner_matrix.py
test/unit/ or test/integration/ config tests
```

Actions:

1. Add generic `base:` resolution with deep merge and cycle detection.
2. Add `suites` to `ExperimentConfig` with schema validation.
3. Add CLI `--suite`; preserve `--level`, `--seed`, and all old configs.
4. Resolve each suite into a run-specific config without mutating the shared
   config object.
5. Keep `obstacle_levels` as the default path when `suites` is absent.
6. Reject duplicate suite names and illegal budget changes.

Acceptance:

- Existing representative configs dry-run with identical resolved values.
- Existing runner integration tests pass.
- A synthetic two-suite config produces four runs for two seeds.
- Base inheritance produces the same dictionary as a self-contained config.
- A cyclic base reference and duplicate suite name fail clearly.

### P2 — Add contact-task environment adapters

Files primarily affected:

```text
genedynamics/experiments/plugins/environments/manipulator_surface_scan.py
genedynamics/experiments/plugins/environments/manipulator_peg_insert.py
genedynamics/experiments/plugins/environments/humanoid_box_push.py
genedynamics/experiments/plugins/environments/__init__.py
genedynamics/experiments/plugins/obstacles/ (empty/no-obstacle adapter if needed)
```

Actions:

1. Wrap the existing environment factories; do not copy task physics.
2. Implement the task-owned structured reset hook.
3. Support model and hidden/OOD execution environments.
4. Expose state/action dimensions and position extraction only where meaningful.
5. Register the plugins in the unified runner.
6. Add smoke tests for all surface families, Peg suites, and all four H1 task resets.

Acceptance:

- Each plugin creates and resets its task through the unified registry.
- Surface medium/geometry and Peg execution-OOD overrides resolve correctly.
- Model and execution envs share the intended robot/action interface.
- No dummy obstacle or placeholder robot is introduced.

### P3 — Add independent receding-contact method adapters

Files primarily affected:

```text
genedynamics/experiments/plugins/methods/
genedynamics/experiments/plugins/methods/__init__.py
genedynamics/experiments/runner.py
```

Actions:

1. Add independent plugins for MGA, Model-based Only, standalone RL,
   DIAL, MPPI, PegasusFlow, ISSA, and ATACOM.
2. Reuse existing solver factories and `run_receding`; do not duplicate solver
   update rules in the plugin.
3. Refactor only the minimum factory boundary needed to accept already-created
   model/execution environments.
4. Return the actual collected state/action trajectory and per-step infos.
5. Record the resolved MGA component contract for every task.
6. Enforce the task-level fairness budget before execution.

Acceptance:

- Every algorithm is discoverable by its own method-plugin name.
- DIAL/MPPI/PegasusFlow/ISSA/ATACOM are not routed through an MGA variant flag.
- One-step and short receding rollouts run for all three tasks.
- MGA fails validation if a required prior/reliability/component is missing.
- Baselines cannot silently fall back to a different controller.

### P4 — Make unified results correct for contact trajectories

Files primarily affected:

```text
genedynamics/experiments/framework/experiment.py
genedynamics/experiments/plugins/metrics/extractors.py
genedynamics/experiments/runner.py
```

Actions:

1. Accept a plugin-returned `Trajectory` directly, preserving structured states.
2. Pass actual `x0`, seed, execution env, timing, and infos to metric plugins.
3. Register the PegInsert metric plugin.
4. Save standard per-seed `results.json` and trajectory artifacts.
5. Extend summary aggregation to nested general-metric dictionaries.
6. Sanitize NaN/Inf with explicit missing counts.
7. Save suite metadata and resolved component/checkpoint metadata.

Acceptance:

- Metrics computed from collected states match direct task extractor calls.
- Peg/H1 metrics do not change when replay would have bifurcated.
- Standard result paths and filenames match existing paper experiments.
- `overall_summary.json` contains task metrics rather than only timing.
- Results are strict JSON.

### P5 — Prove legacy/new execution equivalence

Run one seed per task through both the legacy harness and the new unified path:

```text
surface scan: rigid_cylinder, MGA and DIAL
PegInsert: id_wide, MGA and DIAL
H1: unjamming, model-based MGA and DIAL
```

Compare:

- resolved env/method/budget config;
- reset state;
- executed action sequence;
- task signal series;
- headline metrics;
- prior/revalidation/emergency diagnostics;
- runtime synchronization policy.

Acceptance:

- Deterministic action/state series are equal within a documented numerical
  tolerance, or every difference is traced to output-only state collection.
- Headline metrics match within tolerance.
- No reward, safety threshold, fallback, or checkpoint differs.
- The new runner produces standard outputs and the legacy outputs remain
  unchanged.

If equivalence fails, stop and fix the adapter. Do not tune the algorithm to
make the new output look better.

#### P0--P5 implementation record (2026-08-24)

- P0 minimum freeze: Docker/package/checkpoint hashes, selected evidence
  inventory, and immutable legacy-file hashes are recorded in
  `docs/mga/freeze/p0_minimum_freeze.json`. This record is part of the scoped
  implementation snapshot; no dirty run is promoted to formal evidence.
- P1: base inheritance, deep merge, cycle detection, suite expansion, paired
  budget validation, and backward compatibility are covered by unit tests. All
  18 canonical algorithm configs resolve and dry-run, while the representative
  legacy `single_2d/mbd.yaml` protocol remains unchanged.
- P2: all 13 surface suites, 3 PegInsert suites, and 6 H1 suites create and
  reset through the unified environment registry; hidden execution/OOD
  environments preserve the robot observation/action interface.
- P3: eight algorithm-level method names dispatch their own registered
  controller. MGA rejects a missing learned-component contract, and
  baselines cannot silently route through an MGA flag variant.
- P4: contact metrics consume collected structured execution states. Standard
  seed results contain compact executed q/qd, controls, infos, nested task
  metrics, component provenance, and receding diagnostics; nested summaries
  retain count/mean/std/missing fields under strict JSON.
- P5: all six configured legacy/unified cases have identical resolved
  configs and bit-identical executed action arrays. PegInsert matches 21 DIAL
  and 65 MGA metric/diagnostic scalars; H1 matches 35 DIAL and 32
  model-based scalars. The surface legacy harness replayed actions rather than
  retaining executed states. Explicit replay of the unified DIAL and MGA
  actions reproduces all 30 legacy task metrics exactly (maximum absolute
  difference 0); formal metrics intentionally use the collected trajectory.
  Runtime is synchronized after every execution step and reported separately,
  not used as a deterministic-equivalence requirement. Exact hashes and action
  shapes are stored in the P5 section of the freeze file.

### P6 — Create canonical configs and paper scripts

Actions:

1. Create the three canonical config trees from the validated legacy values.
2. Put all surface suites in the shared surface-scan list.
3. Put one algorithm in each YAML.
4. Remove device/version terms from canonical names and paths.
5. Create paper orchestration scripts that call only the unified runner and
   shared experiment report/render utilities.
6. Add a config audit that compares budgets, seeds, physics, checkpoints, and
   success/safety definitions across algorithms.

Acceptance:

- Every YAML passes unified-runner dry-run.
- Every algorithm sees the same suite physics and paired seeds.
- Learned-method budgets/checkpoints pass the fairness audit.
- No formal paper script references the legacy MGA runner.
- No formal script uses semantic environment variables to alter the protocol.

Implementation record (2026-08-24):

- The canonical surface tree contains the eight algorithm configs, MGA,
  and four causal surface ablations. Each ablation inherits MGA and its
  controller flags differ by only the named mechanism.
- `runner.py` accepts plural seed/suite selections in one invocation so the
  manifest and aggregate summary cannot be overwritten by a sequence of
  single-seed runs.
- `--development-root` mirrors canonical project-relative output paths under an
  isolated root and records `run_class: development` plus the canonical path.
  It creates no development YAML tree and cannot alter formal output paths.
- The formal surface script and shared verifier run all four causal ablations
  on all thirteen suites.

### P7 — Run a two-seed integration matrix

Use development seeds, not formal evaluation seeds, for this gate.

Run P7 directly through the unified runner with `--development-root` and
`--seeds 0 1`. Use a root outside the repository, for example
`/private/tmp/enerdynamics-mga-p7`; do not call `scripts/paper/mga/run_all.sh`
and do not write development seeds into canonical result directories. Select a
whole development subset with one `--suites ...` invocation rather than
separate calls that would replace the protocol manifest and aggregate summary.

Surface scan:

- Run every rigid/soft geometry for MGA and DIAL.
- Specifically confirm that `convex` now completes.
- Run all three hybrid maps for MGA and the key geometry/retraction ablations.

PegInsert:

- Run ID/PoseOOD/SensingOOD for MGA, DIAL, Model-based Only, and
  standalone RL.

H1:

- Run Force Regulation, Fixed-Stance Push, Unjamming, and Walk-and-Push for
  DIAL and Model-based Only.
- Run MGA only if its checkpoint/component contract is complete.

Acceptance:

- No suite crashes or produces missing/NaN headline metrics.
- Result layout, summaries, PNGs, and GIFs are generated automatically.
- The direction of the validated legacy results is reproduced.
- Convex/hybrid failures are resolved before formal seeds are touched.

Execution record (2026-08-25):

- P7 ran only under the external development root
  `/private/tmp/enerdynamics-mga-p7`; canonical paper result directories were
  not modified. The completed matrix contains 74 runs: 26 Surface, 24
  PegInsert, and 24 H1. Every run has `results.json`, an executed trajectory,
  PNG, and GIF, and every selected config passes the visual-aware verifier.
- The unified runner now supports `--resume`. It reuses a result only when the
  saved per-run execution contract matches, while treating seeds/suites/levels
  as dispatch dimensions. The final full-matrix invocation still rewrites one
  complete manifest and aggregate summary. This was required to isolate
  high-memory PegInsert seeds after Docker Desktop VM interruptions without
  accepting stale results or replacing complete summaries.
- PegInsert passes the mechanism gate. Across ID, PoseOOD, and SensingOOD,
  MGA achieved `safe_insertion_success = 1.0` and zero force/torque
  violations for both development seeds. DIAL, Model-based Only, and standalone
  RL retained high raw insertion success but only 0--0.5 safe success per suite,
  with higher peak lateral force. This validates reporting safe success rather
  than raw success alone.
- H1 supports the model-based geometry narrative without claiming a nonexistent
  learned MGA contract. Model-based Only removes the DIAL 30 N force spike,
  wins Fixed-Stance Push nominal and Unjamming safe success, and ties DIAL at
  1.0 safe success on Walk-and-Push. DIAL is stronger on the two-seed
  Fixed-Stance Push OOD success rate. Force Regulation is a force-step test,
  so its zero task `safe_success` is not interpreted as failure; force peak,
  tracking error, balance, and violation metrics are the relevant endpoints.
- The overall P7 promotion gate does **not** pass yet. Surface rigid convex is
  healthy, and MGA improves safety/tracking, but soft/hybrid progress is
  below DIAL and the no-stiffness ablation improves hybrid progress. Therefore
  the current Log-SPD stiffness contribution is not causally validated on the
  hybrid suites. Do not start formal seeds 10--19 or P8 checkpoint locking until
  this Surface mechanism blocker is resolved and the P7 subset is rerun.

Surface blocker resolution (2026-08-25, supersedes the preceding Surface
conclusion but not its historical result record):

- The original hybrid map lived on normalized `xi in [0,1]`, while the locked
  100-step scan only visited approximately `[0.1000,0.1495]`; therefore the
  purported spatial-material suites were locally constant. The three canonical
  hybrid suites now map that finite scan segment over the material chart and
  use a smooth transition. The integration probe observes both the 2 kN/m and
  8 kN/m regions without increasing `scan_rate`.
- Hybrid states lie outside the frozen cylinder reliability checkpoint support.
  Surface MGA now treats this as learned-model abstention and delegates to
  its task-owned model certificate; the default OOD veto remains unchanged for
  tasks without that contract. Learned reliability is therefore **not** claimed
  as validated by the hybrid result and must still be addressed in P8.
- Surface acceptance now makes only physical force-bound violation
  lexicographic. Contact retention, deformation, and force-target error remain
  declared score/Pareto quantities rather than being required to improve in
  every coordinate against a shifted incumbent. This removed the seed-dependent
  safe-incumbent freeze while retaining final candidate revalidation and the
  task-owned zero-force emergency.
- The controllability gate attenuates the empirical response lead, keeps clean
  tangential geometry active, and selectively gates unreliable
  normal/stiffness/force projection and retraction. This reduced the worst Full
  hybrid force from 52.90 N to 44.01 N without reducing completion. Executed
  physical SPD stiffness, rather than ignored chart coordinates, is used by the
  stiffness metrics and ablations.
- The final external development matrix is rooted at
  `/private/tmp/enerdynamics-mga-surface-stiffness-v13`; formal seeds and
  canonical result paths were untouched. Across 2 seeds x 3 hybrid maps, Full
  MGA has mean path coverage 0.9951, realized completion 1.0, common force MAE
  8.996 N, zero contact loss, zero force violation, deformation CVaR95 0.01046,
  force CVaR95 37.04 N, and worst realized force 44.01 N.
- Against the paired no-stiffness ablation under the identical final gate,
  Full changes mean path coverage by +0.0611, force MAE by -3.711 N,
  deformation CVaR95 by -0.00325, force CVaR95 by -8.97 N, and peak overshoot
  by -11.30 N. Both methods have zero force violation. Against the existing
  no-controllability-geometry matrix, Full improves mean coverage by 0.0264,
  force MAE by 0.302 N, deformation CVaR95 by 0.00084, force CVaR95 by
  3.75 N, and peak overshoot by 4.83 N.
- One-seed convex guards remain healthy: soft-convex coverage rises from the
  old P7 value 0.703 to 0.931 with zero contact loss/violation; rigid-convex
  coverage changes from 0.941 to 0.911, while selected-candidate safety rises
  from 0.92 to 1.0 and force violation remains zero. This small rigid coverage
  regression must remain visible in later formal reporting.
- The Surface hybrid mechanism blocker is resolved for development promotion.
  Forty-nine focused unit/regression tests and the finite-map/controllability
  Docker probes pass. P8 checkpoint/calibration locking is still required
  before formal seeds 10--19; this record does not authorize skipping P8.

### P8 — Lock learned checkpoints and causal ablations

Actions:

1. Fix per-task training domains, interaction budget, training seeds, model
   architecture, and checkpoint selection rule.
2. Retrain any unfair learned baseline, including the current scanning ATACOM
   budget mismatch.
3. Verify MGA and standalone RL load the exact same task checkpoint.
4. Train/freeze learned reliability using training/development data only.
5. Freeze PegInsert `no_rl_prior` and `no_learned_reliability` configs before
   viewing their formal-seed results.
6. Establish the H1 MGA prior/reliability/safety contract and the
   suite-aware checkpoint resolver required to run all eight algorithms.

Acceptance:

- Every checkpoint has a hash, protocol, training seed, step count, and domain
  list.
- No evaluation seed enters training, checkpoint selection, or calibration.
- Peg ablations differ from MGA by exactly the named mechanism.
- The H1 method label accurately reflects the active components, and all eight
  algorithms pass a one-step and one-seed shape/interface smoke on every suite.

#### P8 execution record and frozen decision (2026-08-26)

P8 is complete.  Formal seeds 10--19 were not used for policy training,
checkpoint selection, reliability fitting, or conformal calibration.  The
task-base YAMLs now carry a machine-checked `learned_component_lock`; the
configuration audit verifies checkpoint existence and SHA-256, the complete
training protocol, seed disjointness, shared Full/standalone policy binding,
and the H1 evidence label before a formal run can start.

The frozen learned components are:

| Task/component | Training budget and split | Selection rule | Frozen checkpoint SHA-256 |
|---|---|---|---|
| Surface PPO prior / standalone RL | 2,000,000 environment steps; policy seed 0; four seen rigid/soft domains | maximum training-time development realized progress; selected at 1,000,000 steps | `763683288fb7fd8dd2a6714a8c15ab129e8c372c01d6c37f33942278bdd518d6` |
| Surface ATACOM | 2,000,000 environment steps; policy seed 0; the same four seen domains and PPO architecture | the same development metric; selected at 500,000 steps | `3807c1e44253d02f03f47ab9f97c4bab6bb16237d7e6521043598a93b2d4a8ea` |
| Surface reliability | 198 seed-0 development transitions for fit; 198 disjoint seed-1 transitions for q95 calibration; rigid/soft convex only | one frozen ridge + split-conformal fit, with no evaluation-based model selection | `359beaa65602a1acf8287f7cc491ba762b61ff527dba60e9e9d28fdc0bb0e2fe` |
| PegInsert PPO prior / standalone RL | 200,000 environment steps; policy seed 0; wide/tight/two OOD training domains | development safe-success Pareto selection; selected at 100,000 steps | `76dd7c1cf95283a6fcf7b4c9b4bb79d4c25e7bfcdcd70e844b89f5ac4d72b161` |
| PegInsert ATACOM | 200,000 environment steps; policy seed 0; the identical four-domain/budget contract | development safe-success Pareto selection; selected at 50,000 steps | `c7a75f8279fc68a50936652928b151262b5ff71c3ab28c4ea99b5c562e9db88f` |
| PegInsert reliability | 1,078 transitions from development seeds 100/101; 539 disjoint calibration transitions from seed 102 | one frozen ridge-logistic + split-conformal fit | `54b7e822b05bb5cbcedc90ef379ffe85bcbb410b68a7d2922b0c05e94a9f14fa` |

The Surface ATACOM budget mismatch is resolved rather than waived.  The old
checkpoint contained 200,000 training steps.  Its replacement completed the
full 2M interaction budget on the same four domains as the PPO prior.  The
training-time development rewards at 0/0.5M/1M/1.5M/2M were respectively
11986.5, 13345.0, 11506.3, 12444.9, and 12014.7, so the preregistered selector
retained the 500k snapshot after finishing all 2M interactions.  The old 200k
binary is retained only as the external rollback artifact
`/private/tmp/mga_p8_surface_atacom_200k_old_seed0.pkl`; no formal YAML points
to it.

The reliability split produced an additional discriminating result.  A broad
Surface fit that included final Full and no-stiffness hybrid trajectories met
q95 coverage but made the gate over-conservative: on development seeds 2/3,
mean hybrid coverage fell to 0.9406 and acceptance to 1%, despite improving
force tails.  This checkpoint was rejected.  The frozen contract learns only
on seen rigid/soft data and treats hybrid material switches as OOD.  Its
calibration upper coverage is 98.99% for force violation, 99.49% for contact
loss, and 96.46% for both deformation and normalized force MAE, with 98.99%
in-support calibration rate.  On the independent hybrid development seeds
2/3 it abstains on 99.3% of replans, so the task-owned model-based sequence
certificate decides rather than an extrapolating learned model.

The final six-run Surface hybrid guard (three suites by seeds 2/3) obtains:

| Metric | Frozen P8 MGA | P7 development reference |
|---|---:|---:|
| mean trajectory coverage | 1.0000 | 0.9951 |
| mean common force MAE | 9.226 N | 8.996 N |
| contact-loss rate | 0.0% | 0.0% |
| force-violation rate | 0.0% | 0.0% |
| deformation CVaR95 | 0.01023 | 0.01046 |
| force CVaR95 | 38.85 N | 37.04 N |
| selected sequence revalidated safe | 100% | 100% |

Thus the checkpoint lock preserves complete hybrid coverage and zero observed
violations/contact loss.  The +0.23 N force-MAE and +1.81 N force-CVaR changes
from the old 200k ATACOM proposal remain visible and must be re-estimated on
the paired formal seeds; P8 does not convert this development guard into a
superiority claim.

PegInsert now has exactly two formal learned-component ablations.  The
`no_rl_prior` config changes the current controllable-gate method by only
`use_rl_prior` and loads no policy checkpoint.  The
`no_learned_reliability` config keeps the identical solver/sampling contract,
loads the shared PPO prior, and removes only the learned reliability model.
The unified-runner adapter records both switches in `component_contract` and
rejects contradictory checkpoint bindings.

H1 remains explicitly `model_based_only` evidence in this historical P8
snapshot.  There is no validated H1 PPO prior or learned reliability checkpoint
and therefore no `mga` H1 YAML in that snapshot.

The final verification consists of 24 formal configs with zero audit errors,
52 focused unit/config/report tests, and 15 unified-runner component-contract
tests.  All pass for the then-frozen Surface/Peg and model-based-only H1 scope.

#### P8 scope extension (2026-08-27; supersedes H1 model-based-only scope)

The formal protocol now requires the same eight algorithm rows in every
environment.  Surface and Peg checkpoint locks above remain valid, but P8 is
reopened for H1 and P9 is **not authorized** until this extension passes.

The H1 learned-artifact minimum is:

- raw PPO for the three fixed-base tasks (`action_size=12`), shared by MGA,
  standalone RL, and ISSA;
- raw PPO, Walk-and-Push walk (`action_size=23`), shared by the same three methods;
- ATACOM tangent PPO for Force Regulation/Fixed-Stance Push (`action_size=5`), Unjamming (`action_size=1`), and Walk-and-Push
  (`action_size=22`);
- one fixed-dimensional H1 reliability checkpoint, or a preregistered
  fixed-stance/walk pair if the single-contract development test fails.

Before training, H1 must expose and test the contracts its algorithms actually
consume: ATACOM equality/inequality dimensions, ISSA safety index, learned
reliability features and risks, and MGA measured-state sequence safety,
revalidation, and task-owned emergency behavior.  The runner must resolve the
correct checkpoint per suite and record the effective component contract and
hash.  Copying Surface/Peg YAMLs or attaching an incompatible checkpoint does
not satisfy this gate.

The H1 `no_tangent` method must switch only tangent projection inside the
current MGA controllability/gate contract; the legacy `mga_no_tangent`
method is not a valid substitute.  Likewise, MBO provenance must report the
effective absence of policy and reliability checkpoints even if its internal
controller is constructed through the MGA sampler.  A `no_mb_rollout`
ablation is not in the frozen matrix because the current registry flag is inert
by design.  Full versus standalone RL/MBO may be reported as a system-level
decomposition, but not as a single-factor rollout ablation.  Adding that causal
claim later requires a genuine no-rollout algorithm and a preregistered extra
30 H1 runs.

#### P8 H1 extension execution record (2026-08-27)

**Corrective audit, 2026-09-09 — H1 promotion gate reopened.** The following
record documents the earlier execution gate, not a current validation of the
learned prior or genuine locomotion. The completed H1 results remain archived
in place; Surface and PegInsert configurations, checkpoints, and results are
outside this repair scope.

- The H1 development comparison uses an additive RL proposal and explicit zero-tail incumbent
  shifting, with emergency-tail preservation. Loading a prior no longer
  changes the Gaussian incumbent's shift convention. Reliability support
  abstention uses the configured model-based certificate instead of silently
  defaulting to an OOD veto. Other tasks retain the legacy solver defaults.
  Here `Nsample=64` counts Gaussian refinement samples; the deterministic RL
  proposal plus eight stochastic expert horizons add nine certifications.
  Report this extra work and runtime; equal `Nsample` alone is not an
  equal-total-rollout claim.
  The first Unjamming seed-0 comparison failed: yaw error improved from 0.0724 to
  0.0128 rad, but final box error worsened from 0.0127 to 0.1604 m; balance
  and non-hand-contact violation rates were 25% and 21%, versus zero before.
  Safe success remained zero. The unvalidated three YAML overrides were
  consequently withdrawn from the canonical MGA config. Their exact values
  remain in the development result's `config_snapshot`; they are not a
  promoted repair. The explicit shift option and default-preservation tests
  remain available for controlled comparisons after prior retraining.
- H1 reliability fitting must pair the **pre-decision state and 17-action
  candidate** with the corresponding future risk window. The earlier
  post-state/single-action fit did not match deployment. The initial diagnostic
  collection used seeds 110/111; subsequent reliability training/calibration
  must use separate seeds 101/102, reserving 110/111 for paired evaluation.
  Duplicate trajectories and overlapping risk
  windows are recorded and do not establish independent 95% test coverage.
- The frozen PPO's deployment loading matches training, but the training
  audit found absorbing task memory surviving Brax auto-reset, missing
  heavy-DR exposure in the initial fixed domain batch, and divergent KL.
  Those checkpoints cannot substantiate the quality of the RL prior. Repair
  training reset/domain coverage and pass stability checks before replacing
  any frozen checkpoint; all H1 methods sharing a changed policy must be
  reevaluated together.
- Legacy Walk-and-Push terminates on box position alone. For DIAL seed 0, the box moves
  about 29 cm while the support center moves about **1 cm backward**. These
  results are short-push evidence, not walk-and-push evidence. The opt-in
  locomotion contract requires body/support advance, forward swing-and-land
  events for both feet, and goal dwell. Gait direction, phase-aligned reward,
  and stance-only foot constraints are under development validation.
- The first 300-step, zero-action locomotion probe failed to walk and fell;
  tightening success criteria alone is not a controller repair. Keep the
  canonical Walk-and-Push YAML unchanged until a physical walking probe succeeds.
  The unified runner now supports a per-suite `n_steps` (e.g. 300 for Walk-and-Push),
  with same-suite cross-method budget checks and stale-result rejection.

Repair validation status: 94 merged regression tests pass, plus the H1
locomotion/support-load/reset-observation/emergency/gait contracts. The first
two zero-action walking probes fell at 2.84 s and 1.16 s, both with zero
valid forward landings. The measured-support-load probe also failed: it fell
after 118 steps (2.36 s), with zero valid forward landings, maximum support
advance 0.054 m, and peak hand force 108.24 N. Its 0.856 m box displacement
is not a walking success. True locomotion remains unvalidated. Canonical
task YAMLs and frozen checkpoints are unchanged.
The six-run reliability development fit is diagnostic only: binary Brier
scores are 0.2292/0.2372 and Unjamming/Walk-and-Push train/calibration trajectories repeat.
It has not replaced the frozen reliability model. The repaired PPO trainer
now obtains task parameters/durations from the canonical YAML and rejects
missing domain coverage or divergent exports; fresh training is recorded below.

All repair probes belong under `results/_development/humanoid_mga_repair/`.
Do not mix them with the archived matrix or claim an MGA performance gain
before paired development comparisons and the subsequent frozen rerun.

**Authorized repair order (2026-09-09): training/implementation checks →
two-seed development comparison → freeze → affected formal reruns.** A
completed job or passing unit tests alone cannot advance this gate.

1. **Repair and training health.** Keep all artifacts under the development
   root above. Use training seed 101; reserve seeds 110/111 for paired
   development evaluation, and keep formal seeds 0--9 unchanged. First run
   a 4096-step fixed-stance PPO smoke through the existing task trainer.
   Require complete reset of task memory, observed coverage of every
   training domain, finite updates/parameters, matching training/deployment
   observation preprocessing (including its standard-deviation floor when
   normalization is enabled), and non-divergent KL. Inspect deployed actions
   for the old constant saturation failure. Repeat the relevant checks for
   ATACOM's training schemas; a healthy smoke is not a useful-prior claim.
   Retrain reliability only from the matching pre-state/candidate/future-risk
   contract; report duplicate trajectories, effective calibration units,
   prediction error, support abstention, and continuous-risk coverage.
   Calibration trajectories must be separate from seeds 110/111. Walk-and-Push needs
   a physically valid gait/controller probe before a new walking policy is
   trained; legacy Walk-and-Push smoke cannot satisfy that requirement.
2. **Paired development gate.** Once the preceding checks pass, compare MGA,
   no RL prior, Model-based Only, and DIAL on Fixed-Stance Push-OOD/Unjamming/Walk-and-Push with seeds 110/111
   (24 evaluations). Verify the Force Regulation 15/30 N force-control regression separately
   before promotion. Keep reward, safety thresholds, low-level controller,
   suite duration, and evaluation seeds identical across compared methods;
   record all extra prior/certification rollouts and wall time. Require MGA's
   safe-success count not to regress against no RL prior or Model-based Only
   in any tested suite. Require at least one repeatable paired benefit in
   success, force-tail cost, or completion time, without hiding a safety
   regression; report losses and ties as well. Unjamming must meet position and yaw
   jointly. Walk-and-Push must meet actual forward landings, body/support advance, box
   goal and dwell—not box motion alone. Inspect trajectories/GIFs and
   emergency/revalidation diagnostics. Two seeds are a development filter,
   not a statistically established paper-level advantage.
3. **Freeze only after acceptance.** Lock the accepted task/controller,
   algorithm parameters, policy/reliability hashes and evaluation budgets.
   Preserve original results and provenance; do not silently replace an old
   run with a new task or checkpoint via `--resume`. Surface/PegInsert
   configs, checkpoints and archived results remain outside this repair.
4. **Formal reruns are conditional.** The H1 matrix is 580 evaluations
   (480 main + 100 targeted ablations). Replacing the shared H1 PPO,
   reliability, ATACOM policies and Walk-and-Push task affects 380 evaluations: 100
   Walk-and-Push evaluations, plus the three fixed-base tasks' 200 learned-method runs and 80 ablations. The other
   200 policy-free baseline runs for the three fixed-base tasks are reusable only after execution
   equivalence and provenance checks. If shared dynamics/controller changes
   invalidate that equivalence, rerun the additional affected cases too.
   Do not enqueue formal work while either the MGA or walking gate fails.

   The read-only reuse audit finds clean `dd8a4b` provenance for all 200
   archived policy-free runs for the three fixed-base tasks. MBO's 50 runs use `mga_base` without
   an effective prior/reliability model or receding-incumbent override;
   this path does not call the changed emergency candidate. DIAL likewise
   bypasses that branch; MPPI/PegasusFlow use independent backends. This
   supports the conditional reuse scope, but does not replace old/new
   execution-equivalence checks. The current rerun range remains 380--580,
   not an unconditional claim that all 200 archives are reusable.

Training check execution: the first fixed-stance smoke (seed 101, requested
4096 steps, `fixed_smoke_seed101`) completed both 2048-step PPO stages but
was **rejected before policy export**. Force Regulation/Fixed-Stance Push covered all four domains and
ended with KL 0.1346; Unjamming's last logged episode-window KL was 0.0243, but
the full-stage KL was 3105.7422. The last window must not conceal the
stage-wide failure. Its Orbax stage checkpoints remain diagnostic artifacts,
not accepted policies. Source inspection of the installed Brax trainer shows
that the default PPO path updates normalization between behavior sampling
and SGD; the existing adaptive-KL path keeps normalization consistent within
an update. Validate that H1-only training path before another export. The
shared learning implementation, arm training, and frozen policies are not
changed by this diagnosis.

The second smoke (`fixed_adaptive_kl_smoke_seed101`) tested Brax's existing
`ADAPTIVE_KL` update path with two SGD passes and a stricter KL rejection
limit of 1.0. It also **failed**, at the first 1000-step progress callback:
episode KL 11776.75 and mean learning rate 0.00833. The installed adaptive
schedule has a hard-coded maximum of 0.01, not a configurable H1-specific
cap. The normalizer-order mismatch is a verified implementation issue, but
not an empirically established sole explanation of the failed training.
Do not promote this adaptive setting or increase the acceptance threshold.
The next controlled alternative is existing fixed-rate PPO (`1e-4`) with
online normalization disabled, using the task's raw observations and the
same serialized train/deploy setting. Keep two SGD passes, gradient clipping,
domain/reset checks and the KL gate. No alternate observation wrapper,
shared-library patch or task-dynamics change is required for this test.
Seven training-contract tests and eight reliability-contract tests passed
before this second smoke; passing contracts did not predict training success.

The third smoke (`fixed_raw_observation_smoke_seed101`) **passed the training
health gate**, completing 4096 steps in 542.45 s. Force Regulation/Fixed-Stance Push and Unjamming final KL were
0.000279 and 0.069182, respectively, with finite parameters, actual coverage
of every stage domain and a constant learning rate of 0.0001. Its checkpoint
SHA256 is `d60d779453bcb6177e5d8a07b400c0100783dac9925e8662e1079ff501aa21c0`.
Online normalization is disabled consistently in training and deployment;
the old normalizer-floor CLI is explicitly recorded as unused by this policy.
This artifact is tagged `pipeline_smoke` / `performance_validated=false`.
It must not be used as the full-budget prior in the paired or formal matrix.
Deployment-action sanity checks also passed on the existing development
observation records: Fixed-Stance Push-OOD saturation fell from 100% (all 12 old policy
channels constant) to 0--0.58%, with no constant channels; Unjamming saturation was
0.50%. This is an observation/action check, not a new rollout or performance
comparison, and the duplicated Unjamming records are not independent trials.
The subsequent 200k fixed-policy attempt (seed 101, eight environments, same
fixed learning rate and KL gate) **failed at 103k observed steps**: Force Regulation/Fixed-Stance Push
completed 100k, but Unjamming reached episode KL 1.2591 at 3k, above the unchanged
limit of 1.0. No `fixed_ppo_seed101.pkl` was exported. Its failure and complete
logged history remain under `_policies/_training/fixed_ppo_seed101/`.
Before this failure, first-stage median episode return already worsened from
about -1231 to -11473; the change in episode length does not explain this.
No development-matrix or formal run is authorized by smoke success alone.

Source audit found that the CPU training setting used one-step unrolls and
Brax's default discount of 0.9. At 50 Hz this has a short effective horizon,
and one-step GAE relies entirely on the critic for longer-term credit. The
next controlled intervention changes only unroll length to 10 and discount
to 0.99, retaining the optimizer, reward, actor, normalization and KL gate.
With eight environments this also changes each update batch from 8 to 80
transitions and reduces SGD counts at the same interaction budget; record
that coupled change. The 16x16 architecture describes the actor only; Brax's
default 256x5 critic is unchanged. First check the actual eight-environment
setting for 4800 steps (2400 per stage), including memory use. Do not use
`--smoke` to claim eight-environment feasibility: that option selects four
environments for the fixed schema. This short check is not a promoted prior;
only a subsequent full-budget artifact can enter performance validation.
The latest combined lightweight regression suite passes all 106 tests.
The temporal-credit change subsequently passed the seven H1 training-contract
tests and the real eight-environment, 4800-step check: both stages completed
2400 steps, with final KL 0.000124 / 0.024167, all domains observed and finite
parameters. Wall time was 625.53 s, with sampled memory below the container's
12 GiB limit (the samples are not a peak-memory measurement). Its unpromoted
checkpoint is `_policies/fixed_temporal_credit_check_seed101.pkl`, SHA256
`52122932360351f19e6b30c6a38779685852c7eaed839abdf39c6154d45d196c`.
Its observation-only deployment check is also finite and nonconstant on the
existing Fixed-Stance Push-OOD/Unjamming development records: Fixed-Stance Push-OOD action saturation is 0--0.58%
(old policy 100%, all 12 channels constant), and Unjamming saturation is 0.33%.
This reuses archived observations, not new physics; duplicated Unjamming records
remain duplicates and these numbers are not a performance comparison.
The same setting subsequently **completed 200k steps** (100k per stage) in
865.34 s, exporting `_policies/fixed_temporal_credit_ppo_seed101.pkl`, SHA256
`8ddd5f6ea6fd20e7263beee206f843839059b357016ec2d121c693013f7491e5`.
Final stage KL was 0.000123 / 0.058508; parameters/domain coverage passed the
existing health gate. All recorded startup source hashes still matched at
completion audit. Within each stage, the first/last ten logged reward-window
medians were -1467/-1023 (Force Regulation/Fixed-Stance Push) and -6608/-5051 (Unjamming); this is a training
trend, not a closed-loop evaluation or cross-run causal comparison.
The full-budget checkpoint's observation-only inference is finite and
nonconstant, with Fixed-Stance Push-OOD saturation 0--0.41% and Unjamming 0.25% on the same archived
development observations. It still has `performance_validated=false`.
The next fixed-prior check uses the existing unified runner on Fixed-Stance Push-OOD/Unjamming and
seeds 101/102 (four new standalone-RL trajectories), isolated under
`fixed_prior_validation/`. Seed 101 overlaps training and these are pipeline/
closed-loop diagnostics, not the held-out paired 110/111 matrix. No comparable
old-policy 101/102 results currently exist; archived 110/111 runs cannot stand
in for them. Neither a successful export nor finite actions satisfy the
paired performance gate.

All four fixed-prior closed-loop diagnostics subsequently completed and passed
the persisted-trajectory, checkpoint and source-hash checks. Fixed-Stance Push-OOD seeds
101/102 reached the line (goal errors 2.48/1.79 mm; force peaks 31.45/42.82 N;
no force-limit events), but both had safe success zero: non-hand collision
rates were 21%/19% and balance-violation rates 11%/7%. Unjamming produced identical
trajectories for both seeds: progress 3.23%, goal error 29.03 mm, yaw error
0.08168 rad, force peak 63.49 N and force-violation rate 1%; task/safe success
were both zero. These are standalone-prior diagnostics, not new full-MGA
results; the duplicate Unjamming trajectories do not provide independent evidence.
The training/inference repair is real, but useful and safe closed-loop prior
performance has not been established. Fresh sequence-reliability collection
uses canonical MBO and MPPI, Fixed-Stance Push-OOD/Unjamming, training seed 101 and calibration seed
102 (eight trajectories), under `reliability_collection/`. Fitting must inspect
class coverage, duplicate sources and the small number of independent
trajectories before any paired MGA evaluation or promotion.
The task-owned reliability trainer now refuses an existing output before data
collection/fitting. Its development artifacts explicitly carry
`performance_validated=false` / `promotion_eligible=false`, source and unique
trajectory counts, missing-class/duplicate warnings, and calibration Brier
against a constant predictor using the training event frequency. Eleven H1
reliability-contract tests pass. Neither this audit nor a high overlapping-
window calibration coverage establishes independent candidate safety.

The released-hand native walking diagnostics also remain below acceptance.
With CPG disabled, the standing probe falls forward at 3.42 s; enabling the
current gait instead falls backward at 2.36 s. Both produce zero valid forward
landings. Matched-state bilateral ankle pulses establish the restoring COM
direction: +0.1 rad gives a -0.194 m/s^2 acceleration change at 20 ms and a
-4.84 mm displacement change at 200 ms. Pitch alone has a different response
and must not be used to flip the feedback sign. A single predeclared no-gait
test will compare per-ankle support gains 375/45 against 250/20 for six seconds,
leaving every other bias/reference unchanged. This tests a gain-margin
hypothesis from the mass and measured short-horizon response; it is not a
stability proof, a gain sweep, an MJX validation, or a complete pushing task.
That predeclared standing test completed 300 steps (6 s) without falling:
final body displacement 0.02490 m, support-center displacement -0.000338 m,
pitch 0.11489 rad, and left/right ground loads 256.4/248.2 N. It took 89.33 s
on the single-CPU native screen. This is evidence of improved isolated support
stability, not locomotion. The next bounded B test enables the unchanged gait
with exactly these two gains; no other bias or task parameter is adjusted.
That B test failed at 109 steps (2.18 s), with body displacement -0.651 m,
support-center displacement 0.118 m and zero valid forward landings. Static
support gains alone therefore do not repair the current gait. Keep these
development controller settings out of the canonical task until the dynamic
support/swing contract is validated.
In B at 1.20 s, the airborne right foot still participates in the legacy
support-center mean, moving the reference about 7.8 cm forward relative to
the loaded left foot and adding approximately -29 Nm per ankle. The same
support correction also acts on the airborne ankle. The next matched test
uses the already implemented `support_phase_foot_level` mode with the same
375/45 gains, rather than further gain tuning. Planned stance and actual
landing can disagree, so removing this error does not itself prove that the
remaining support/swing timing will work. It is queued after the 4800-step
training check; canonical settings remain unchanged.
The matched support-phase test also failed, now at 74 steps (1.48 s), with
body displacement -0.645 m and zero valid landings. Its planned stance / actual
contact allocation and total feedback authority require further diagnosis;
the airborne-foot error is real, but its removal has not established a
working gait. No walking policy training or formal Walk-and-Push rollout is started.
The completed fixed-stance temporal-credit 200k training did not use these
opt-in walking settings and cannot validate Walk-and-Push.
Read-only diagnosis identifies earlier mode/reference errors than the final
fall: at 0.40 s both feet carry 203/324 N, but planned-phase weights are [0,1],
halving the configured total support gain. At 0.58--0.60 s the still-loaded
left foot's ankle target jumps from -0.044 to -0.600 rad as the planned phase
changes; applied torque jumps from +21.50 to -36.95 Nm. At 1.28 s the left
foot still carries 222 N while both support weights are zero. Gait capture
also still uses the two-foot mean instead of actual support, but that later
error must not be claimed as the cause of the initial instability. The next
controller repair must align support allocation with measured contact,
preserve total feedback authority and avoid reference jumps on loaded feet,
then recheck isolated standing and gait before adding the pushing load.
No such untested controller change is made during the current fixed-policy
training: its startup source hashes must remain interpretable.

After the fixed-policy training completed and its source hashes were checked,
the next isolated Walk-and-Push candidate was implemented in the existing controller/env
hooks. It uses measured foot-floor normal loads, not planned swing flags:
with `d=max(F_L+F_R, m_robot*g)`, ankle weights are `2 F_i/d`, and the shared
capture/balance-control reference is the load-weighted foot reference with
the unloaded fraction blended toward the pelvis. This preserves total design
authority at normal/full support and fades it continuously during unloading;
robot weight excludes the separately rooted box. The selected candidate is
the existing `support_phase`, **not** `support_phase_foot_level`, so the original
continuous CPG ankle reference is retained. Legacy and fixed-stance paths
remain unchanged. Array contracts and a same-pre-control-state Native/MJX
normal-load comparison were added to the existing integration test file.
The single 375/45 `gait_measured_support` native probe is queued after the four
fixed-prior rollouts; no extra gain/CPG-amplitude search is queued.

That measured-support probe completed its diagnostic execution but failed the
walking gate at 121 steps (2.42 s): body displacement -0.664 m, support-center
displacement -0.0119 m and zero valid forward landings. The same-state
Native/MJX foot-load comparison passed (maximum absolute difference
4.84e-5 N). Correct contact-load allocation therefore did not by itself make
the open-loop CPG feasible. The remaining diagnosis concerns planned swing
versus actual landing and foot orientation, not another support-gain sweep.
The measured-support trace shows the old spurious ankle saturation is absent
at 1.40 s, but the right foot has not landed reliably while the clock starts
flexing the still-loaded left knee again. The right foot subsequently makes
heel contact at a large pitch angle. The unloaded diagnostic also retains
`stance_hip_bias=stance_ankle_bias=-0.2`, together introducing a nominal -0.4 rad
foot-pitch bias. The next queued check sets only those two biases to zero,
keeping support gains 375/45: first a six-second released-hand standing
screen, then gait only if standing completes without a fall, final body drift
is at most 5 cm, maximum support-center drift is at most 1 cm, both final foot
loads are at least 10 N, and the load-parity check passes. These are bounded
diagnostic gates, not formal walking criteria. Do not combine this reference
test with an untested phase-clock change or promote its overrides to YAML.
The eight-run reliability collection and development fit precede these probes
in the CPU queue; no full-MGA comparison or formal matrix is queued yet.

**2026-09-09 queue revision after the fixed-stance failure audit.** Do not
execute the fit above automatically. The four MBO collections are complete;
MPPI Fixed-Stance Push-OOD is finishing, with the remaining collection/fit held for a targeted
controller check. Fixed-stance WBC still braces against commanded force even
when measured hand contact is zero (the strict Walk-and-Push path already uses measured
load). In MBO Fixed-Stance Push-OOD seed 102, hand contact is absent at steps 12--41, while
the independent pre-clipping ankle brace term rises from -19.60 to -101.25 Nm
per ankle; non-hand collision starts at step 42. This is a temporal/mechanistic
diagnosis, not yet proof that this term alone causes the failure. Replay the
complete MJX state to step 12 and compare the same recorded action suffix
under the original WBC and its existing `support_load=min(command,measured)`
hook, without changing actions or task thresholds. Use raw-Unjamming step 35, where
command and measurement nearly agree, as a one-step negative control. Preserve
all collected records; if the low-level dynamics changes, do not present an
old-controller reliability fit as calibrated for the repaired controller.
The isolated Walk-and-Push neutral-reference checks remain next; no formal freeze or
performance acceptance follows from these diagnostics alone.

The matched-state intervention subsequently completed with strict 1e-5
archival replay checks and equal full-state hashes at both branch points.
The first two attempts stopped at a numerical replay check because they JIT-
compiled reset, unlike the contact-task runner's eager reset; these attempts
did not test the physical intervention. The completed artifact is
`measured_support_counterfactual_20260909T152606259336Z/results.json` below the
H1 repair development root. For the Fixed-Stance Push-OOD seed102 42-step suffix, non-hand
force peak fell from 1408.27 N to zero, and body displacement from 0.29287 m
to 0.06956 m. Both branches failed task success; the measured-load branch
made essentially no box progress. In the Unjamming one-step negative control, where
command and measured load were nearly equal, body displacement differed by
0.62 micrometres and box progress was unchanged. This supports repairing
false load compensation, not claiming improved closed-loop success.

The existing task-owned `support_load` hook is now used for fixed stance as
well as strict locomotion. The legacy box-only Walk-and-Push path remains unchanged.
No reward, force threshold, contact target or hand-impedance equation changes.
Recheck closed-loop Force Regulation/Fixed-Stance Push/Unjamming before retraining; do not immediately fit the old
collection or declare the repaired task solved from a fixed-action suffix.

The collection checkpoint is now six verified trajectories: four MBO and two
MPPI Fixed-Stance Push-OOD runs. Both post-save interruptions were in the temporary queue's
validator (`level` in newly returned rows versus `suite` in saved JSON), not
missing simulations. Their 100 actions, 101 states, task signals and source
hashes were checked; do not rerun or overwrite them. The remaining two MPPI
Unjamming collections and reliability fit are held pending the controller diagnosis.

The neutral-reference standing A failed at 74 steps (1.48 s), with body drift
-0.796 m and pitch -0.861 rad, so its dependent gait B was not run. Removing
both stance biases is not an accepted repair. At home the actual robot COM
is 3.13 cm behind the foot-site center; at 0.20 s the pelvis-based and actual
COM feedback errors already have opposite signs while both feet remain
loaded and nearly level. The next isolated standing comparison retains the
failed A's neutral references and 375/45 gains, changing only its feedback
point to robot COM. Verify position/velocity against native `mj_subtreeVel`,
excluding the separately rooted box. Do not zero out the physical COM error
by subtracting its home value. This is a diagnostic controller intervention,
not a new task setting or a walking-success claim.

That true-COM standing comparison also failed: 109 steps (2.18 s), body drift
-0.782 m and pitch -0.833 rad. COM-velocity parity passed with maximum error
1.37e-9 m/s, including independence from a changed box velocity. Correcting
the feedback point alone is therefore insufficient; do not promote this
runtime-only proxy or start its dependent walking test. Diagnose the remaining
posture/COM modes before selecting a further intervention.

**Walk-and-Push execution revision: reference the vendored DIAL task first.** Stop the
isolated neutral-reference/controller-gain sequence here. Failure of a zero-
action CPG is not evidence that closed-loop planning or learned feedback cannot
walk. The vendored `UnitreeH1PushCrateEnv` optimizes 19 absolute joint targets;
its gait is a reward reference, not a prescribed joint CPG. Our 23-dimensional
Walk-and-Push instead exposes a 12-dimensional contact primitive plus only 11 bounded
CPG residuals. At reset the hip/knee/ankle target intervals are
[-0.7,-0.5]/[0.7,0.9]/[-0.7,-0.5] rad, versus DIAL's
[-1,1]/[0,1.74]/[-0.6,0.4]. Larger action dimension does not establish equal
whole-body control authority.

Use `baselines/dial-mpc/dial_mpc/examples/unitree_h1_push_crate.yaml` as the
reference: 300 steps, 2048 samples, Hsample=24, Hnode=6, 4 diffusion rounds
(10 initially), dt=timestep=0.02 s, slow_walk, and velocity ramp to 0.8 m/s
over 2 s. It has no goal-line success predicate or force-control safety
certificate. Its loop executes the previous first action before shifting
and replanning. Do not equate this demonstration with our strict safe
walk-and-push success or with a 100-step/64-sample run.

1. Audit the original assets and hard-coded contact indices under the current
   runtime, then run a reduced-budget development reference through the
   existing experiment runner, delegating environment/planning to vendored
   code. Explicitly record the sample/seed deviations. The project legacy
   PushCrate alias is not a valid substitute yet: its shared robot asset has
   disabled hand collision masks without the explicit hand pairs present in
   the contact-control scene. Do not restore masks in the shared asset and
   thereby change the force-control tasks.
2. Evaluate an opt-in task/controller realization with the existing 12 hand
   coordinates unchanged and the last 11 coordinates mapped to robot-owned
   joint-target ranges. This is an algorithm-independent Walk-and-Push interface, not a
   special permission for MGA. A reward/initialization gait reference must
   not silently overwrite the optimized joints.
3. Account for initialization, receding tails and emergency actions in the
   new coordinates. Zero is a joint-range midpoint, not a safe stance. Recheck
   physical support, actual landings and force metrics in closed-loop tests;
   no zero-action gait-success prerequisite is imposed on the planner.
4. Revalidate/retrain Walk-and-Push policy and reliability components if this realization
   is adopted. An unchanged 23-dimensional shape does not make checkpoints
   semantically compatible. Keep canonical YAML, formal results and the other
   two tasks unchanged until the new development gates pass.

The reset-only contact audit is saved as
`dial_vendor_contact_index_audit.json` under the repair development root.
With JAX 0.6.2, Brax 0.14.1 and MuJoCo 3.6.0, the vendor's 28 contact slots
retain the expected hand-box indices 26/27 and foot-floor indices 2/3/6/7.
The project legacy alias has only 24 slots: disabled hand masks remove both
hand-box candidates, and JAX's out-of-range gather returns torso-box contacts
for the old wanted-contact indices. This defect concerns the native legacy
alias, **not** the eight-algorithm paper DIAL adapter, which already runs the
shared `humanoid_box_push` task with explicit hand-box pairs. The audit ran
zero physics steps; it validates contact identity, not a successful rollout.

The reduced-budget **original vendor reference has now completed** at
`dial_vendor_reference_seed110_n64/`: 300 executed steps, Nsample=64 (not the
original 2048), otherwise the original H24/node6/4+10 diffusion schedule and
original execute-shift-replan loop. It took 217.9 s, moved the box 3.159 m,
pelvis 3.394 m and mean foot position 2.858 m. Saved contact/foot trajectories
also show alternating lifted forward landings, not just sliding. However,
peak hand/nonhand-box normal force was 528.1/655.3 N. Its original loop
continued through 263 done states, all caused by crossing task sampling
joint bounds, not torso-height/flip termination. These bounds are not the
physical joint limits. This is evidence for direct-joint planning authority,
**not** a force-safe Walk-and-Push success or a formal comparison result. The standard
trajectory GIF is a render of saved q/qd, with no physics replay.

The opt-in `joint_target` realization is implemented in the existing robot
profile, WBC and task. It uses the vendor's affine mapping for the 11 leg/waist
coordinates, leaves the 12 hand-impedance coordinates unchanged, and removes
the CPG/bias/brace/COM-feedback additions from direct joint targets. Only
`leg_grav_comp=0` matches the vendor's leg-PD formula; arm control and 4 ms x 5
force-loop integration remain task-owned. The optional receding initializer
and task emergency candidate inverse-map current joints; range midpoint zero
is not interpreted as a hold command. An out-of-range pose is clipped to a
representable target, not certified safe. The gait still selects nominal
manifold stance rows, although it no longer prescribes low-level joint motion.

Only this new mode exposes reward-phase sin/cos (H1 observation 76 -> 78) and
a JSON policy-interface contract. Old residual-action checkpoints are rejected
even if action width matches. The existing H1 trainer can receive explicit
development Walk-and-Push overrides and an episode length; no new YAML or runner is used.
Six array contracts and a real-H1 reset/interface check passed (the latter
executed zero physics steps). The optional receding-initializer tests passed
14 cases. The latest five-file unit suite passed 110 tests, including all
eight new trainer/checkpoint-interface cases; these eight are not additional
to the 110. These interface tests do not establish walking or safety.

The first new-task DIAL closed-loop check completed under
`joint_target_closed_loop/`, seed 110, 300 steps, N64/H24/node6/4+10 and
temperature 0.05. Relative to canonical Walk-and-Push, task overrides are exactly
`joint_target`, `locomotion`, leg gravity compensation 0, `slow_walk`, legacy
gait reward reference, and a 0.50 m push distance. Other rewards, the 60 N hand
limit, box dynamics and strict landing/success thresholds remain unchanged.
The source hashes are recorded and unchanged. Runtime was 1136.6 s (967.8 s
planning). It **failed**: zero valid forward steps, task/ safe success both 0,
first recorded fall at step 110, peak hand force 88.18 N at step 73. Box
progress 0.947 m is not successful 0.5 m push-line tracking; body progress
after the fall is not evidence of walking. Canonical configuration and formal
artifacts have not been promoted.

The saved trajectory isolates a load/following mismatch. At step 50 the box
has moved 0.499 m, while body/support displacement is -0.012/-0.263 m. Box
speed reaches 0.975 m/s, versus the 0.25 m/s body target. During the no-hand-force
interval, deceleration is 1.00 m/s^2, consistent with 8 N sliding resistance
and 8 kg mass. At taper onset, passive stopping distance is about 0.406 m but
only 0.118 m remains. The approach force floor is 10.5 N, still above the 8 N
resistance. The step-73 impact occurs after the force reference has reached
zero; it cannot be fixed merely by zeroing a positive force command. The
vendor's 30 kg/50 N load delays box motion, but copying 50 N resistance while
retaining a 30 N tracking target would introduce another incompatible demand.

Pre-register one load contrast, not a reward sweep: 8 versus 27 N sliding
resistance, all other physical/reward/safety/planner settings fixed. The
27 N hypothesis follows the approximately 13.425 J work of the existing
ideal 30 N distance-taper profile over 0.5 m (26.85 N mean resisting load).
This ideal one-dimensional calculation is not a success prediction. Because
the acquisition bug below is being repaired, both arms of this contrast must
use the same corrected source and seed 110; comparing an old-source 8 N run
with a new-source 27 N run would not isolate load. Persist actual foot loads,
floor clearances, subtree COM, torso pose and box velocity in the existing
metrics pipeline before these runs. Do not change the 60 N safety limit,
silently tune reward weights, or promote formal results from this contrast.

For fixed stance, the first closed-loop measured-support regression is also
complete at `measured_support/fixed_stance_regression/`: model-based-only Unjamming,
seed 101, 100 steps/N64/H16. It reaches the position/yaw goal at step 46
(3.572 mm position error, 0.02604 rad yaw error), with no fall, nonhand-box
collision or balance violation. A single 64.64 N impact at step 19 exceeds
the unchanged 60 N limit, so task success is 1 but safe success is **0**.
The post-success trajectory is absorbing, not 54 additional physical steps.
This repairs progress but does not pass the safety gate; no claim of MGA
dominance or permission for formal reruns follows from it.

Two matched-action Unjamming diagnostics are complete under `measured_support/`.
`acquisition_reference_counterfactual_20260909T162720974575Z/` replays both
100-action branches from the same eager reset; original replay error is
exactly zero. Replacing only the interpolated acquisition reference with
the physical face target delays the latch from step 1 to the first real
contact at step 32, but changes the peak from 64.64 to 66.56 N. The old
post-release action sequence is no longer appropriate to this altered
trajectory, so its failed task result is not a repaired-controller ranking.
It proves the early latch is a reference-contract bug, not that fixing it
alone eliminates the impact. The permanent correction must use finite
physical geometry rather than an infinite plane or an interpolated target.
Current MJX sphere-convex positive distances can be the sentinel 1.0, so
contact slots alone are not a continuous near-contact distance sensor.

`release_chart_counterfactual_20260909T163456141436Z/` compares ten actions
from the identical full state at step 16. Keeping the old off-centre `a=0`
instead of the original recenter increases the peak from 64.64 to 115.63 N
and produces yaw overshoot. Initial leg/waist torques and force commands
are identical; only hand targets/arm torques change. Do not remove recenter
merely because its target shift is large. Both probes ran unchanged source,
and neither substitutes for a newly planned closed-loop validation.

The permanent H1 acquisition correction now uses the existing `BoxObstacle`
signed distance in the actual box-geometry frame and subtracts each actual
hand-sphere radius, after domain-randomized geometry has been installed.
Both finite, nonpenetrating hand gaps must lie in the existing 20 mm approach
band, or the existing measured-force trigger must fire. The non-sphere/G1
fallback is unchanged and is not covered by the H1 claim. Eight focused
acquisition/direct-joint tests passed; the real reset check executed no physics
steps. Seven metric-extractor tests passed, and a real Walk-and-Push reset confirmed that
the seven new raw diagnostic signals persist through the full extractor.
The matched load queue at `matched_load/` locks these source hashes until
both 300-step runs finish (both are now complete with unchanged hashes). This is a development mechanism contrast, not a
formal seed sweep.

The corrected-source 8 N run completed in 1140.3 s with unchanged hashes:
task/safe success 0, box progress 0.9466 m, hand peak 88.18 N. The newly
persisted raw signals expose two measurement problems, without turning this
failure into evidence of safety. The actual environment fall predicate first
fires at step 102 (torso 0.57696 m < 0.588 m), whereas the old metric's fixed
0.5 m threshold first fires at step 110. There is also one real right-foot
forward landing at step 64, after backward landings by both feet; the old
site-height counter misses it. A stricter offline event count requiring an
observed unload edge and continuously supported opposite foot yields left 0,
right 1, still a failed Walk-and-Push. Do not describe the motion as either no stepping
at all or successful forward walking.

Before the first simultaneous loss of foot load (steps 1--69), gait tracking
RMSE is 4.509 cm using foot-site height but 6.298 cm using physical foot-floor
clearance. At unchanged weight 30, the site objective understates that gait
cost by about 49%. At step 64 the right foot carries 303 N with -3.05 mm
clearance while its site is 8.77 cm high, falsely appearing closer to the
14.97 cm swing target. This supports first aligning the measurement with
the vendor's physical foot-floor distance, before adjusting gait weights.
All 600 saved foot loads are finite: 426 exactly zero, 174 positive, minimum
positive 23.754 N. Strict zero is sufficient for this diagnostic; this is not
a sensor-noise calibration. Positive `g_bal` before a later fall is not by
itself a false positive: retain the current risk threshold while repairing
measurement, rather than deleting conservative warnings to improve SSR.

The matched 27 N arm also failed: 1116.6 s wall time, task/safe success 0,
box progress 1.6089 m and peak hand force 206.27 N. Physical fall occurs at
step 86 (1.72 s), earlier than the 8 N arm's step 102. The peak is at step 88,
after falling and after the requested force has tapered to zero. Nevertheless,
the pre-fall peak is already 99.44 N at step 76, against a 19.64 N reference;
terminal bookkeeping cannot remove this real overshoot. Each arm has only
left 0/right 1 valid forward steps before falling. Nonhand-box force is zero
in both. Reject increased resistance as a demonstrated walking fix; do not
promote the larger box displacement or use post-fall movement as performance.

The subsequent strict `joint_target` development realization now measures
gait error and nominal stance-height residuals from physical foot-floor
clearance. The legacy DIAL reward target is unchanged; the CPG alternative
subtracts its nonzero home-site baseline when compared with clearance.
Actual forward-step events require an observed unload edge, opposite-foot
support, the existing 2 cm lift, and a loaded recontact at the existing 5 mm
clearance/4 cm forward-displacement thresholds. Loaded sliding, toe roll,
in-place lifts and full-air intervals do not manufacture forward steps.
Support progress uses recorded landings, not moving site positions.

Strict Walk-and-Push observations append 13 task-memory coordinates (91 total), while
preserving the original 78-dimensional prefix and the 23-dimensional action
interface: swing/load eligibility, relative landing displacement, step counts,
goal dwell and independently saturated startup/acquisition clocks. Other
realizations keep their previous 76/78-dimensional observations. Exact policy
interface checks reject old checkpoints; dimension padding is not retraining.

**Terminal-objective consistency gate.** The successful Unjamming trace above earns
0.81235 on completion but zero on every success-padding step; its previous ten
active rewards average 0.96744. PPO termination cuts bootstrap, so auto-reset
does not credit the next episode to the completing action. This creates a
real incentive mismatch, although feasibility of any particular indefinitely
holding candidate has not been established. Before retraining, preserve the
true completion-step reward and physical risk, then give the absorbing goal
the existing active-reward upper bound `w_alive + w_prog * push_dist`.
Remove AL optimization penalties only on unexecuted success padding, never
on the actual completing transition or its force/risk labels. The H1-only
training wrapper must keep the absorbing goal until the collection cap,
retain time-limit truncation/bootstrap, and immediately reset genuine falls.
This is a continuing discounted absorbing-goal objective, not a finite-horizon
terminal bonus. Count active and padding transitions separately. Contract
implementation was applied only after the matched load pair completed. The
H1-only training-side toy/config/audit suite passed six tests. A subsequent
15-case array suite passed terminal/objective, shared safety-boundary, training
and physical-walking contracts; the additional nonzero CPG home-height case
also passed. These are contract checks, not successful closed-loop trials.
The broad regression found a new observation test fixture incorrectly omitted
the base reset's required `step=0`; only the fixture was corrected. Its final
core rerun passed 149 tests (36 slow cases explicitly deselected), including
the real reset/interface checks for all three contact-task adapters. The
earlier broad attempt was interrupted during the slow tail after 150 passes
and that one fixture failure; it is not an all-green 184-case result.
Four additional reliability semantic/provenance/serialization tests passed.
The H1 loader now rejects old equal-width q95 files without a matching
task-owned risk/feature/horizon/interface contract. New calibration requires
saved collection-time contracts and matching source/asset hashes; historical
signals remain audit-only and are never silently replayed into new labels.
Independent review confirms the success-side contract, but not complete
RL/MPC termination equivalence: PPO still terminates genuine falls immediately,
whereas fixed-length MPC predicts active dynamics/rewards after a fall. The
latched fall prevents subsequent task success but does not zero these predicted
rewards. Audit the post-fall objective before claiming full terminal consistency;
do not silently change it during the source-frozen closed-loop pair.

The next source-frozen closed-loop queue is
`physical_contact_contract/`: first DIAL Walk-and-Push seed 110, 300 steps, N64/H24/node6,
4+10 diffusion, temperature 0.05 and original 8 N resistance; then model-based
only Unjamming seed 101, 100 steps, N64/H16/node4 and the canonical 2+10 schedule.
The Walk-and-Push overrides and reward/safety weights are otherwise those of the matched
8 N arm. The updated source/asset hashes, task-memory and physical safety
signals are recorded. These runs are development checks, not formal seed
selection or a claim that full-MGA has passed its deployment gate.

If corrected-clearance Walk-and-Push still fails, the next prespecified single-factor
contrast is `w_prog: 8 -> 0`, retaining `w_box=5`, the 0.5 m goal and all
locomotion/safety thresholds. A second, separate contrast may replace only
the cold-plan mean with the vendor's affine joint midpoint, retaining the
current execution order and all other settings. The actual vendor trace
executes two initial zero-chart actions; the current adapter executes one
inverse-mapped joint-hold action. Both pin node zero and shift with a zero
tail, so changing initialization alone is not an exact reproduction of the
vendor's execute/shift/replan schedule. Do not combine these contrasts into
one unexplained gain change or copy the vendor's force-unsafe objective.

The corrected-clearance DIAL Walk-and-Push run is complete (1147.2 s, unchanged hashes):
task/SSR 0, goal error 0.6817 m, final box progress 1.1817 m, left/right steps
0/0 and hand peak 54.63 N. Offline reconstruction matches all saved load,
eligibility, swing and step-count states over the 75 pre-fall transitions
exactly; this is not another missed-step report. Both feet are unloaded at
steps 13--20 and land backward at step 21. Later apparent forward landings
also follow full-air intervals and do not satisfy the supported-step contract.
The pre-fall hand peak is 45.82 N at step 11; the larger 54.63 N peak is after
falling. Balance first becomes positive at step 22, the force reference reaches
zero at step 45, the box crosses the line at step 46 and the robot falls at
step 76 (1.52 s). Compared using the same physical definitions, the old 8 N
trace falls at step 102 and has one valid right step: force-limit violations
decreased, but locomotion became worse. No promotion is justified.

The first Unjamming attempt in that queue stopped after its first physical step
because the inline progress logger requested Walk-and-Push-only walk-memory fields.
It produced no complete trajectory/performance result. Its two audit artifacts
are retained in `_failed_attempts/mbo_unjamming_missing_walk_log_fields/`; the corrected
logger uses the extractor's existing zero defaults for fixed stance. Only Unjamming
is retried, with identical runtime hashes, and the single-factor Walk-and-Push progress
contrast waits for successful completion of that queue. A logger error is not
counted as an environment failure or hidden as a successful run.

The retried Unjamming physics and standard result writing completed: 100 actions,
101 states, task/SSR 0, position error 0.24068 m, yaw error 0.07153 rad,
hand peak 67.56 N (two over-limit steps), three balance-violation steps,
no fall and no nonhand-box collision. A final inline validator incorrectly
required Walk-and-Push-only raw foot diagnostics; independent post-save validation of
the common signals, finite states/actions, task metadata, hashes and summaries
passed without another simulation. The diagnostic status retains this error;
the queue's completion means valid data, not a performance pass.

Unjamming's geometric acquisition is at step 36, first measured contact at 42 and
clearance release at 49 (yaw 0.06134). The best yaw is 0.05591 at step 51;
the required 0.03 rad is never reached. Box position alone enters tolerance
at post-steps 72--73, while yaw remains 0.07091--0.07133. The historical
`completion_step=71` field is the box-only zero-based index, not task completion.
The force peak comes later at step 82 and cannot explain the earlier failure
to meet position and yaw together. Saved outer-step integrator endpoints stay
within [-23.15, 27.69] N; do not claim saturation or PI causality from them.

The next scoped Unjamming change, pending the Walk-and-Push source-frozen contrasts, is to remove
only the clean chart's forced lateral-contact `a` equality. Geometric release
at 5 mm clearance must not disable subsequent yaw correction: the current
chart fixes `a` to the centre after release, leaving only vertical `b` free.
Keep the physical primitive, release latch, six stiffness/one force/three face
equalities, reward and all success/safety thresholds. This yields a/b as two
optimization directions (ten clean rows) and requires a new 2D ATACOM tangent
head; it is not the old matched-action intervention that held `a=0` and made
impact worse. The new proposal restores optimizer freedom rather than fixing
another hand-designed contact location. Its performance remains untested, and
the retained 45 N command equality still cannot by itself guarantee braking.

The prespecified `w_prog=0` Walk-and-Push contrast is complete at
`box_progress_contrast/dial_walk_and_push_progress/` (1111.6 s, unchanged runtime hashes).
It still has task/SSR 0 and zero valid forward steps on both feet. First fall
is delayed from step 76 to 104 (1.52 to 2.08 s), and pre-fall peak hand force
decreases from 45.82 to 38.77 N. These are partial improvements, not walking:
at step 50, box progress is 0.313 m while body/support progress is
-0.018/-0.253 m. The final 0.084 m goal error is also misleading in isolation:
the box first overshoots, then moves backwards after the fallen robot passes
it. Full-trajectory peak force is 78.21 N, with seven over-limit transitions.
Do not promote this reward setting or count post-fall body displacement as
locomotion. It supports a contribution of progress shaping to the early rush,
but does not establish the cause of the remaining leg-control failure.

The separate affine-midpoint cold-plan contrast retains `w_prog=8` and all
other corrected-clearance reference settings. Its first invocation failed
before physics because the inline diagnostic tried to assign a read-only
task property. The two incomplete audit artifacts are retained under
`cold_plan_contrast/_failed_attempts/midpoint_readonly_initializer/`.
The retry passes the override through the controller's existing
`initialize_plan` keyword; task source, emergency hold, later shifts and
execution order are unchanged. This is a single initial-mean diagnostic,
not a new canonical configuration or an exact vendor-loop reproduction.

The cold-mean run has now completed all 300 transitions and standard result
writing in 1143.7 s, at unchanged runtime hashes: task/SSR 0, no valid left or
right steps, first balance violation at step 27 and first fall at step 85.
Pre-fall peak hand force is 40.60 N; the full peak is 139.89 N, with two
over-limit transitions after falling. Final box progress is 1.654 m, not a
successful 0.5 m push. Its final inline validator incorrectly demanded
bit-exact zero after spline interpolation: the actual first action's maximum
magnitude is 1.31e-8, while the cached N2U first row has a 4.48e-8 off-diagonal
absolute sum. Independent finite-data/source validation passed using a
float32 numerical tolerance, without resimulation or changing any physical
safety/performance threshold. The diagnostic status retains the original
assertion error. Neither this initialization nor `w_prog=0` is promoted.

After both Walk-and-Push contrasts completed, the scoped Unjamming free-contact patch was
applied in the existing task and tests. Unjamming now has ten clean equalities and
two optimizer-controlled contact coordinates; raw action/observation widths
and Force Regulation/Fixed-Stance Push/Walk-and-Push branches remain unchanged. New contracts cover both sides of
geometric release, actual tangent filtering, rank/dimensions, rejection of
old 1D ATACOM heads and export of a 2D head without changing the shared raw
fixed policy. Physics and regression results must be checked before this
mechanism is considered validated.

All six new contract cases passed in the existing `dev-cpu:local` image
(6.62 s; 92 unrelated cases deselected). The subsequent Unjamming/MBO seed101
100-step N64 closed loop at `free_contact_geometry/mbo_unjamming/` completed in
341.4 s with unchanged hashes, but failed task/SSR: goal error 0.04096 m,
yaw error 0.08085 rad, hand peak 103.92 N, two force violations and four
nonhand-collision transitions. There was no fall. This patch restores a
degree of freedom; it has not demonstrated a performance improvement.

The new optimizer mostly chooses raw `a < -0.5`, reversing the sign of the
rear-face target offset relative to the old off-centre `a=0` choice. Neither
contact coordinate clips. Yaw never improves beyond its pre-contact 0.07575
rad minimum, geometric release never occurs, and the position-only goal band
at post-steps 94--95 still has yaw about 0.080. Nonhand contact first occurs
at step 87; wall force is 168.66 N at step 92, before the step-93 hand peak.
Wall force later reaches 546.53 N. The peak's immediate target displacement
is small, so neither a large same-step coordinate jump nor an outer-step
integral endpoint alone establishes its cause.

Audit candidate scoring before another geometry/weight change. With the
existing AL `2|r|+100r^2`, a 1 cm single-coordinate residual costs 0.03,
whereas correcting yaw 0.07575 to zero gains only 0.02869 per step from its
explicit task term. H1 has spherical hand contactors and translation-only
impedance, so distal-axis alignment is not required merely for sphere-plane
contact. However, the same-state axis residual is independent of lateral
coordinate `a`; any directional preference must be assessed through the
rollout, not asserted from these scale estimates alone. The next diagnostic
compares constant `a=0` and `a=-0.7` over 17 steps from replay-verified full
state 1, with identical other coordinates, physics and AL weights. These
are controlled counterfactual candidates, not the original saved samples.
The first attempt was killed at its 4 GiB Docker memory limit (exit 137;
the Docker `oom` event confirms the cause). It saved no branch comparison.
Its preflight record is retained, marked failed, under
`free_contact_geometry/_failed_attempts/candidate_score_audit_4gib_oom.json`.
The identical physical comparison completed sequentially with 12 GiB in
202.44 s; resource failure is not evidence about the contact objective or
candidate quality. Replayed q, qd and observation errors were exactly zero,
and the final source-hash check passed.

Both 17-step candidates have zero actual hand-box force throughout and
essentially identical final yaw (0.07575 rad) and box displacement. Yet
`a=0` versus `a=-0.7` gives mean raw rewards 0.88047 versus 0.88969 and mean
AL scores -1309.06564 versus -890.89118. Virtual commanded-wrench friction
alone contributes penalties 1307.47819 versus 889.19823; hand-position
penalties are 0.83158 versus 0.57434 and axis penalties 1.61275 versus
1.98468. This demonstrates pre-contact friction scoring dominating these
counterfactual candidates, not that the axis residual is their principal
directional cause. It does not establish safety or improvement of a repaired
closed-loop optimizer. Holding these exact predicted trajectories fixed and
removing only their pre-contact friction charge reverses the two-candidate
ranking: -1.58746 for `a=0` versus -1.69294 for `a=-0.7`. This is an offline
objective decomposition, not a new physical rollout. Next assess a task-owned, current-contact-conditioned
soft objective, keeping the physical constraint and realized safety hooks
distinct and making both rollout-score paths consistent.

The implemented mode-only repair is restricted to H1 `unjam`. It enables
the unchanged newton-valued virtual friction residual only for a current
right-hand/box geom pair with nonpositive separation and positive normal
compression. A left-only contact, a latched approach/acquisition flag, or a
departed/tensile contact is insufficient. The task's sampling AL and sequence
acceptance score share this soft residual; raw constraints, ATACOM's input,
real force/nonhand/balance/fall risk, the ten-row chart, and task physics and
reward remain unchanged. This is a desired-wrench compatibility cost, not a
measurement of actual friction or a new safety certificate. The next Unjamming
seed101/N64/100-step development run compares against the failed free-coordinate
reference above; inspect pre-contact direction, actual contact acquisition,
joint yaw-and-position completion, and force/nonhand violations.
Ten focused contact-score/terminal/safety/chart contracts passed in 6.67 s
(four new cases included; 28 unrelated cases deselected). These are array
contracts, not MJX closed-loop validation. The H1 environment at this freeze
has SHA256 `9bbd2575933eeccd7c9297cc5349dfc51a6df5279311d0cc7197f8557da8aef0`.

The matching `contact_mode_scoring/mbo_unjamming/` closed loop completed in 350.19 s
and remained unsuccessful/unsafe. Importantly, it **did** correct yaw: steps
52--56 satisfy the 0.03-rad limit, reaching signed yaw -0.000365 at step 54.
It then over-rotates to -0.09209 at 80 and finishes at -0.05905. During the
angle-valid interval the box has advanced only 0.70--0.79 mm of its 30 mm
target; position-only entry at step 96 has yaw -0.06628. Neither result meets
the joint completion condition. First physical hand contact changes from
32 to 43, distinct from the new acquisition latch at 28. The geometric
release latch at 46 is not a declaration that yaw is finished; the box
re-enters the corridor wall at 63.

The repaired run has peak hand force 137.60 N at step 85 (old 103.92 N),
one above-limit transition (old two), nonhand collision rate 6% (old 4%),
and no fall. Raw deployed-state friction residuals are nonpositive at
45--84, so their AL contribution is zero: do not attribute this interval's
continued over-rotation to a positive deployed friction cost. Unselected
candidate scores were not saved. The force integral is +30 at 46--84, and
the post-state requested effective force is clipped at 60 N; at 85 the
integral becomes -30, the hand peak and first nonhand contact coincide.
Outer-step data cannot resolve their substep ordering or prove PI causality.

A concrete remaining geometric defect is the bilateral target span. With
half-width 0.55 m and hand spacing offsets +/-0.21 m, the current shared
center can put the left target outside the box face. It is 42.43 mm outside
at 84 and only 7.90 mm inside at 85, following center changes of -139.69 mm
at 83--84 and -50.33 mm at 84--85. Wall loading already precedes this event.
Pre-register the next geometry-only repair: on H1/Unjamming rear-face contact,
parameterize the common center over half-span
`actual_half - abs(hand_offset) - contact_edge_margin = 0.30 m`, retaining
both hands' fixed spacing and 4 cm edge margin. Do not independently clamp
hands or simultaneously tune PI, force, reward, or safety limits. This is a
real physical-chart change (raw `a=0` center becomes 0.15 rather than
0.275 m), not a no-op guard. Preserve Force Regulation/Fixed-Stance Push/Walk-and-Push/G1 exactly and explicitly
leave side/mixed-face semantics outside this first repair's claim.

The implementation audit narrows the face-selection claim further: canonical
Unjamming CFS includes three equality rows on raw face logits, so final refined
commands are near-rear, not a validated learned side-face selector. Raw
samples are scored before that retraction and can still use mixed faces.
Neither the existing right-hand residual nor the four physical-risk channels
certify both targets' finite-face membership. The span correction must not
introduce an all-face geometry-safe label. Its next single-case physical
check is `rear_contact_span/mbo_unjamming`, seed101,100 steps,Nsample64,Hsample16,
Hnode4,2+10 updates, temperature0.06, using the existing unified runner and
unchanged model-based-only algorithm. Compare against `contact_mode_scoring`
under the same H1 metric contract. Inspect actual hand/box contact, the joint
position-plus-yaw success condition, force/collision/fall risk, the complete
yaw trajectory and saved face weights. Merely reducing a peak or briefly
crossing zero yaw is not task success. No checkpoint or formal run is selected
by this development check.

The rear-span implementation and its regression checks are now in place:
16 selected cases passed in15.34 s (13 integration/array-constructor checks
and3 prior-interface checks), with AST and diff checks clean. The constructor
compares actual half-extent against the sum of hand offset and edge margin;
the zero-span0.25 m boundary is rejected without a tolerance that would mask
invalid geometry. Force Regulation/Fixed-Stance Push/Walk-and-Push/G1 target arithmetic matches the independent old
formula exactly; Unjamming retains76D observations,12D actions,10 clean equalities
and2 ATACOM tangent coordinates. These tests do not execute a physical
rollout or establish side/mixed-face safety. H1 environment source SHA256 is
`fb9165a65578844a467b3ad9776324438bf5545680ce5416a8a37698d7926945`.
The single100-step rear-span physical check then completed in358.82 s with
all standard artifacts, finite states and an unchanged source hash. For the
first time in these matched model-based-only Unjamming checks, actual task success
occurs on transition67 (zero-based completion metric66): position error
0.003207 m and absolute yaw0.009550 rad. The67 active transitions have no
fall, balance violation or nonhand collision. However, hand force peaks at
166.876 N on transition50 and exceeds60 N on3/67 active transitions (4.478%);
force CVaR95 is109.647 N. Safe success remains zero. Preserve all100 raw rows,
including33 explicitly marked unexecuted success-padding rows, but do not
use padding to dilute risk. This is a task-completion improvement, not a
safety/performance promotion. Unjamming's current completion rule is position plus
yaw without fall; release is a separately reported latch, and there is no
Walk-and-Push-style velocity/dwell condition. Completion therefore does not establish
a stationary or sustained post-goal recovery. Next inspect the pre-impact
target/contact/PI sequence and the common active interval with the previous
case before changing any force-loop mechanism.

The matched-prefix analysis confirms both the gain and its limitation. By67,
the old mode-only case has moved just1.027 mm and over-rotated to-0.09005 rad;
the new span has moved26.793 mm with yaw+0.009550 rad. Its release latch is
one and clearance34.674 mm at completion. New targets keep both hands inside
the rear face with the intended4 cm margin; reconstructed logits remain
near-zero/rear-dominant. Neither run has nonhand/balance/fall violations in
that common67-step prefix, so do not overstate improvements by comparing
different-length active tails. Old prefix force peak is41.675 N with no
over-limit steps, whereas the new prefix has the three violations above.
The completed box still has qvel_x about0.209 m/s and yaw rate-0.267 rad/s;
its frozen tail must not be presented as a measured natural stop.

New force peaks occur at48/50/54 (156.326/166.876/70.109 N), while both
semantic targets are valid and wall/nonhand force are zero. The integral
never reaches its positive30 limit. At48, the beginning-state integral is
-25.755 and the feedforward-plus-PI contribution is only about0.392 N; the
target shifts laterally2.30 mm. At50 that contribution is32.876 N and the
lateral target shift35.52 mm. Thus positive windup, edge re-entry and renewed
wall jamming are not necessary explanations for these new impacts. These
numbers are not total Cartesian wrench or actual substep actuator torque.
Force alternates sharply over47--54 (29.84,156.33,4.39,166.88,0,45.28,16.92,
70.11 N), settling near45 N only later. A retrospectively selected settled
window cannot replace the full active-history risk report.

Pre-register the next minimal observation-only replay: reconstruct actual
State47 by executing the saved reset/action prefix under the same source,
verify its saved q/qd/obs, then instrument the20 physical4 ms substeps of
the original actions48--51. Compare the instrumented transition with ordinary
`step`; preserve all gains, reward, timing and thresholds. Record both hands'
actual normal loads, finite-box gaps, normal velocities, separate positional/
damping/feedforward wrench contributions, integral and actual actuator
limits. This is a contact/force-loop causality check, not an optimization
sweep, new baseline or claim that the current model-based gate already
predicts these spikes. The saved run contains no candidate node horizons,
so its executed actions alone cannot recover the historical MPC predictions.

This replay has completed under the archived runtime source. The first
attempt stopped before the suffix because its integrity hash included
`str(PyTreeDef)`, whose MJX auxiliary-array wrappers contain unstable object
addresses. Preserve that failed attempt. The corrected hash canonically
encodes node types, actual static auxiliary arrays and dynamic leaves;
synthetic sensitivity tests and ten repeated reset hashes passed without
physics. The `hash_stable_retry` then passed all prefix/archive, full-state
ordinary/instrumented and source/input integrity checks:55 outer steps,
275 physics steps,20 instrumented substeps,121.05 s. Archive and instrumented
state errors are zero; the independent diagnostic torque recomputation has
maximum error3.815e-6, below the unchanged1e-5 tolerance.

The measured substep peak is331.503 N at49.2, larger than the166.876 N maximum
visible at outer-step endpoints. At49.1 the post-state force is1.564 N; the
next applied feedforward-plus-PI contribution is34.206 N, followed by the
331.503 N impact. The subsequent contribution clips to-60 N and contact force
drops to zero. Concurrently, per-ankle brace contributions alternate between
approximately0 and-41 to-52 Nm, and actual ankle torques alternate between
approximately+20 and-22 to-32 Nm. This establishes high-frequency coupled
contact/force/support oscillation, not a causal attribution to PI or brace
alone. Separate them with matched-state, unchanged-action interventions.
The integral itself stays negative throughout this window; it is the total
feedforward-plus-PI contribution that repeatedly changes sign. Neither arm
nor leg torques saturate, arm Jacobian condition numbers stay below2.557,
and positional wrench magnitudes stay below2.508 N per hand. Do not describe
this as positive integral windup, excessive nominal stiffness or a singular
arm. Pre-register two separate20-substep interventions from the same State47:
hold the integral at its actual anchor value, or hold only the additive
two-ankle brace torque at its anchor value. Preserve the other feedback paths,
all four saved actions and original clipping. A reduced peak with lost motion
is not a closed-loop repair, and two effective interventions would not identify
a unique cause. Recheck the ordinary archived prefix after any Walk-and-Push-only source
edit; record that source difference explicitly.
Both interventions completed in162.32 s with295 physics steps and all
archive/anchor/control/source gates passing. Baseline / held integral / held
brace peaks are331.50 /81.54 /144.76 N, with8 /6 /5 above-limit substeps and
force-limit excess impulses3.86471 /0.30963 /1.03276 N s. Corresponding box
advances are0.5767 /0.8382 /0.1805 mm and yaw changes-7.724 /-9.892 /-6.879 mrad.
Thus holding the integral strongly attenuates the local transient without
reducing this window's forward/yaw correction; holding brace also attenuates
it but reduces forward motion. Both remain unsafe. These observations support
two contributing feedback channels, not a unique root cause, a useful constant
integral controller or full-episode validation. All comparisons preserve the
original60 N limit and actual initial integral rather than clearing it.
Outer-endpoint reporting and risk checks must not be represented as physical
substep safety; audit an H1-owned substep envelope before promoting a safe
MGA result. Keep the original endpoint results and failed attempts intact.

The H1-only substep collection and evaluation upgrade is now implemented in
the existing environment, extractor and report utility. Each fast-force step
retains its actual post-physics hand force, nonhand force and four signed
safety margins, with an explicit coverage flag. Control, reward, observation
width, AL and the action chart are unchanged. A fall observed inside an
executed interval now latches failure even if the endpoint recovers: this is
an explicitly stricter termination contract, not just extra logging. Unknown
active coverage cannot certify safety or produce a valid reliability label.
Reset and unexecuted padding do not claim fresh physical samples; G1 retains
its endpoint contract, and both arm tasks retain their original behavior.

H1 evaluation/reliability contracts now identify the substep semantics. Keep
the old20 ms force curve and endpoint diagnostics, but bind the new force
safety primary to `physics_force_peak`, `physics_force_violation_rate` and
`physics_force_normalized_cvar95`; the last is CVaR95 of actual substep F/f_max.
Impulse uses the4 ms right-rectangle rule, without adding duration for the
initial state. Initial risk participates separately in safe success. Report
plots select endpoint and physical fields exactly and write separate files;
missing physical evidence never falls back to an endpoint primary. Twelve
environment array/reset contracts and104 metric/prior/report unit cases
passed; the latter took13.97 s. These are contract checks, not a new closed-loop
performance result. H1's hard gate now uses the same strict zero signed-margin
boundary as physical safe success; an additional regression rejects a5e-9
H1 balance violation while preserving G1's legacy tolerance.

The real archived-impact integration subsequently passed in138.76 s, with
275 physical steps and unchanged source/input hashes. All52 archived states
including reset match in their old fields, and the20 hand/nonhand force samples
match the independent substep audit exactly. The standard metric plugin reads
the actual states without calling reset or step: it recovers331.503 N physical
peak and166.876 N endpoint peak, with full sampling coverage. A separate branch
from the same full State47 changes only `ki_force` from1 to0.08. Physical peak
falls to85.210 N, nCVaR95 from5.52504 to1.42016, physical violation rate from40%
to35%, and force-limit excess impulse from3.86471 to0.345748 N s. Box advance
increases from0.5767 to0.9139 mm and yaw correction from7.724 to9.655 mrad.
Every outer interval nevertheless contains an unsafe substep, and both
branches are correctly rejected. This is a local feedback-timescale result,
not a safe controller or evidence that the reduced gain establishes45 N from
reset. The next isolated Unjamming check must use a complete reset-to-goal rollout;
do not combine that gain change with the next Walk-and-Push objective test.

That full-reset Unjamming check subsequently passed on development seed101 in375.51 s
(`integral_rate/mbo_unjamming/validation.json`). The only configuration intervention
is `ki_force: 1 -> 0.08`; the solver remains the archived `mga_base` development
controller, not full MGA or the canonical no-RL-prior ablation. All100 actions
and101 states are retained: completion is State67/1.34 s, followed by33
unexecuted padding steps. All335 real substeps have known coverage and no
force, nonhand, balance or fall violations under the declared physical
contract. Task success and safe success are both1. Physical peak is44.0857 N
and nCVaR95 is0.732334; endpoint peak falls from166.876 to43.990 N. The old
global physical peak is unknown, with the earlier replay establishing only
the lower bound331.503 N; do not present that bound as a full-episode maximum.

This is not a failure to establish force: during the13 full-reference outer
steps, the actual65 physical samples average43.6098 N against45 N, with
MAE1.3902 N. The integral remains within[-7.3033,6.3502], without hitting its
30 N clamp. Completion time is unchanged from the baseline; final goal/yaw
errors are3.619 mm/0.011795 rad versus3.207 mm/0.009550 rad. Thus the single
seed trades a small in-tolerance terminal error increase for a large measured
force improvement, without delaying task completion. Runtime inertia is
confirmed unchanged at[0.1,0.1,0.1] kg m^2, mass8 kg and half-extents0.55 m.
The virtual wrench-based friction-cone diagnostic still flags52.24% of outer
samples: the physical SSR pass does not certify every AL/manifold residual.
Preserve source/input hashes and this distinction. A second development seed,
Force Regulation/Fixed-Stance Push retention and the separate actual-unload contract are still required
before task-wide gain promotion or full-MGA training/acceptance claims.

The separate physical UNLOAD support-path comparison is complete at
`rear_contact_span/unload_support_path_state47_48/diagnostic.json`. It uses
the original `ki_force=1`, the same actual State47 and unchanged stiffness,
with fixed world targets2 cm outward from the measured hand centers on their
unique finite box faces. Both branches remove positive normal feedforward;
only whole-body support differs: zero support load versus the measured
remaining hand load, capped at45 N. Over the next20 ms, physical peaks are
57.18 and31.79 N respectively, versus156.33 N for ordinary continuation.
Both branches satisfy initial-state and all five substep physical checks;
the first arm wrench and torque are identical, so this is a support-path
comparison, not an arm stiffness change. Source/input hashes and the exact
ordinary-prefix replay passed.

Prefer the measured-load branch for the next integration test, not as an
already validated emergency guarantee. At20 ms, both hands still penetrate
the contact geometry and force remains about17.6 N; the positive clearances
of their commanded targets are not achieved hand clearances. Continuous
fixed-anchor unloading, actual retained stiffness, NORMAL re-entry and
invalid-entry behavior remain unvalidated. This fixed-stance support result
does not transfer to Walk-and-Push's direct joint-target path, which has no additive
brace controller. All executed unloading substeps must remain in physical
safety statistics even where the normal-only reliability model abstains.

The UNLOAD execution context is now integrated in the existing H1 task,
MGA backend/facade, receding bridge and unified runner. It seals the actual
entry stiffness matrix/raw coordinates and world-frame hand targets, keeps
commanded normal force and integral nonpositive, and retains measured support
load in the existing fixed-stance controller. Pending and committed modes are
explicit; a nominal rollout cannot replace a sealed execution entry. Invalid
entries are rejected before real execution, while valid but predicted-unsafe
mitigation remains explicitly uncertified. NORMAL-only reliability excludes
UNLOAD windows without removing their measured force or safety samples.

Explicitly rejected attempts retain their actual prefix, failure reason and
failed success outcomes. Prefix force/cost statistics are stored separately,
not pooled as cheap completed episodes. Reports retain attempted/completed
denominators. A preflight guard protects every selected aborted result and
its protocol manifest before any output write, with or without `--resume`.
Success padding cannot open/cancel an unload entry or clip its saved integral;
it retains frozen physical risk and no new executed samples. The existing
physics loop still computes and discards its padding suffix: this is not a
claim of zero simulation work.

Actual CPU Docker regressions passed in three rounds:215 unit cases (42.53 s),
64 selected environment/plugin/runner integration cases (727.58 s), then34
boundary and neighboring cases after the two audit fixes (84.53 s). These
counts overlap and must not be added as unique tests. The integration round
includes canonical Surface/Peg/H1 scene resets, not new formal evaluation.
The next physical test replays the saved ordinary State47 prefix, then checks
30 continuous UNLOAD intervals and one NORMAL re-entry, including direct
callback parity. Its maximum budget is81 outer calls/405 simulated substeps,
with original `ki_force=1`, fixed entry K/targets and no extra metric replay.
Contracts passing does not establish continuous separation or safe re-entry.

That physical continuation is now complete at
`rear_contact_span/unload_execution_context_continuation/diagnostic.json`,
448.91 s, exactly405 simulated substeps, unchanged source/input hashes and
maximum complete-state parity error1.31e-6. The150 actually evaluated UNLOAD
substeps have peak31.7947 N and zero force/nonhand/balance/fall violations.
First zero measured force occurs at48 ms and both geometric gaps are positive
at56 ms, but right-hand contact recurs at540 ms (10.0309 N). At600 ms the
right/left gaps are-18.37 micrometres/+5.774 mm. Thus bounded physical safety
passed, not continuous detachment or zero-force invariance. Actual stiffness
and world targets stay exactly sealed while the pelvis advances59.19 mm and
the box only0.194 mm; this is not a target following the box.

The specified old action48, not an MGA-selected proposal, is then executed in
NORMAL for20 ms. Its five forces are[0,0,0,28.9384,27.5938] N, without physical
violations, but the original KI=1 integral rises from-5.988 to+29.103. Do not
infer safety over the next horizon or a learned/gated planner from this one
action. The combined31-interval diagnostic has full physical coverage,
task success/SSR0 and tracking error against the benchmark45 N; it validates
unloading, not task completion or full-MGA superiority. All150 UNLOAD samples
remain in safety statistics despite normal-only reliability abstention.

Next, freeze one current runtime for five development checks: Unjamming MBO seed102
at KI0.08, Force Regulation 15 N seed101 at KI1/0.08, and Fixed-Stance Push nominal10 cm seed101 at KI1/0.08.
The latter two are genuine same-source, same-seed single-factor pairs. The
earlier successful Unjamming seed101 predates this execution-context integration;
report that source difference instead of calling it an identical-source
two-seed cohort. MBO here is `mga_base`, not `no_rl_prior` or full MGA. Force Regulation
30 N/Fixed-Stance Push-OOD retention, a fresh learned prior/reliability and true receding
MGA gate execution remain outside these five checks.

The fixed-candidate gate-to-execution diagnostic is complete under
`rear_contact_span/gate_bridge_state47_48_memory_retry/`. The first12 GiB
attempt is retained separately as a Docker-confirmed OOM during joint
acceptance compilation; it completed only47 prefix steps and one NORMAL
witness, so it is neither a mechanism pass nor failure. A single14 GiB retry
changed only the exclusive output directory and resource limit and completed
in568.21 s:86 outer-step equivalents/430 logical physics substeps, unchanged
871-source/four-input hashes and exit0.

The real gate revalidates both fixed NORMAL candidates as unsafe (force-risk
component1), applies its force veto, and selects the task-owned UNLOAD candidate
without a task override. UNLOAD is revalidated safe and committed with
execution mode1. The known NORMAL witness reproduces peak156.3263 N; the one
actually executed UNLOAD interval has forces[31.7947,10.0597,16.8215,15.6137,
17.6248] N and zero positive physical safety margins. Predicted and actual
complete states, entry stiffness/target seal and committed shift mode match
exactly. The final attempt SHA256 is
`4b6e4e66154afbb8fa7e4c222900f617306c4bd2267fed38c95d2b5b2af43404`.
This validates the fixed-candidate gate/host-execution mechanism only: neither
candidate was produced by natural search, and no RL prior or learned
reliability was active. Task/strict-safe success remain0 for this one-step
unloading diagnostic; do not report it as full-MGA task performance.

The first of these five checks has completed at
`integral_rate/unjamming_ki0p08_seed102/`, in582.46 s. It has task/strict-safe
success1, completion index68 (69 actual transitions/345 valid4 ms samples),
goal error4.035 mm and yaw error0.028918 rad. Physical peak is44.8293 N and
nCVaR95 is0.735622; force/nonhand/balance/fall violations are all zero with
coverage1. The31 success-padding transitions remain in the100-action tape
but not the physical denominator. All871 pinned source files and three
input artifacts are unchanged. This is a current-runtime MBO feasibility
result, not a fresh full-MGA or same-source two-seed result. The paired Force Regulation/Fixed-Stance Push
checks continue in their preregistered order; no gain or formal configuration
has been promoted.

The Force Regulation 15 N same-seed pair is now complete (KI1:553.50 s; KI0.08:563.10 s),
with identical initial q/qd/observation, unchanged871-source/three-input hashes,
and100 active transitions/500 valid physical samples in each run. Both have
zero force/nonhand/balance/fall violations. Lower KI reduces peak37.8199 to
35.0705 N but increases nCVaR95 from0.276848 to0.315302 and endpoint tracking
MAE from3.07037 to3.32890 N. Under the same saved full-reference mask
(outer50--100,255 physical samples), mean force changes15.00276 to15.08022 N
and MAE0.010839 to0.114619 N. This is a peak/settling tradeoff, not uniform
improvement. Force Regulation's unchanged line-goal success/SSR is0 because its box is fixed;
interpret the force-step and physical-risk data rather than treating that
unreachable position goal as a force-control failure.

The Fixed-Stance Push nominal pair is also complete (KI1:565.20 s; KI0.08 completed in the
same fixed queue). Both reach the10 cm line with strict-safe success1, full
physical coverage and zero force/nonhand/balance/fall violations. KI1 completes
at index36 with1.154 mm error; KI0.08 completes at index39 with0.735 mm error.
Lower KI changes peak47.5342 to49.2021 N, nCVaR95 0.521944 to0.525949 and
whole-active-prefix force MAE14.4896 to12.1827 N. Thus it improves one aggregate
tracking measure while slowing completion and slightly worsening peak/tail
force. Neither run reaches a completed30 N benchmark interval before success,
so no steady30 N statistic is inferred. The two results retain unchanged
871-source/three-input hashes and185/200 actual4 ms samples; success padding
is excluded. Together with the Force Regulation and Unjamming checks, this rejects a global KI0.08
promotion. Keep KI suite-specific only if a later frozen contract and
independent validation justify it.

H1 evaluation was also corrected separately, without re-writing historical
results: persist all raw transitions but omit only the continuous unexecuted
success-padding suffix from physical statistics. Retain the real completing
transition, its force reference, and every actual violation. The primary
`completion_step` follows the task-success event; `box_completion_step` reports
position-only entry separately. Version the H1 evaluation contract and refuse
to silently apply it to unmarked historical signals. The opt-in general metric
view leaves Surface/Peg plugins unchanged. Eight new cases and the complete
86-case extractor/prior/report regression passed (the eight are included in
86; zero skips). This is metric correctness, not improved task performance.
An additional 50 configuration/receding-controller/MGA-backend unit tests
passed in 12.69 s, including the frozen arm-prior shift behavior and the
Surface/Peg ablation configuration contracts. No arm/Peg task configuration,
physics source, checkpoint, or historical result was changed by this pass.

The current fixed-schema PPO training/export health check also completed:
seed101, eight environments, exactly4800 transitions split2400/2400 between
the Force Regulation/Fixed-Stance Push and Unjamming curriculum stages, elapsed443.41 s. The recorded maximum KL
is0.00012979 in stage0 and0.00084324 in stage1, with a finite exported76D/12D
policy. Stage0 includes genuine goal padding; the recorded Unjamming episodes have
100 active steps and no goal completion. The checkpoint is
`_policies/fixed_contact_geometry_check_seed101.pkl` under this development
root, SHA256
`94bedbc2367be78d647e4509b814d5fcf80a5ec5cd8bb7dc1d7f3e10ce238721`.
The actual loader subsequently consumed that hash on352 active pre-action
observations from four saved seed101 development traces (Fixed-Stance Push-OOD and Unjamming,
model-based and historical raw RL). All outputs were finite, none reached
absolute0.98, and maximum absolute action was0.83536. This is an off-policy
numerical audit, not a closed-loop prior-performance result; nominal Force Regulation/Fixed-Stance Push
retention is still untested. An earlier inference-only pass included a
historical Force Regulation seed0 input, was discarded before saving an audit, and is not
used for model selection; the final saved audit uses only development101.
Do not promote this4800-step checkpoint or train the full budget against the
known-invalid Unjamming bilateral chart. Apply and check that physical mapping first.

A separate read-only emergency audit found a real contract gap: in Force Regulation,
`fixed_force_target=True` makes ordinary `_force_cmd` return15/30 N even for
the emergency template's minimum force coordinate. In other H1 tasks a zero
force command can still leave positive integral or positional-impedance
loading. The current candidate therefore cannot be called a verified
zero-force retract. Preserve ordinary Force Regulation force-step semantics; do not infer
an emergency mode from floating-point action patterns. A genuine task-owned
unload transition needs explicit execution context, the same transition in
candidate scoring/execution/replay, bounded retreat, integral handling and
honest initial/substep risk. Its interface is under design, not implemented
or physically validated. This remains a full-MGA promotion blocker and is
independent of the unloaded DIAL walking tests below.

An additional pre-contact check limits the force-mismatch explanation:
in the `w_prog=0` trajectory, hand-box force is zero for the first ten steps,
yet the left foot unloads at steps 4--8 and reloads at step 9 about 10.5 cm
behind its initial location. Maximum clearance is only 2.84 mm, below the
existing 2 cm swing threshold. Arm impedance and predicted future contact
are still active, so this is not an unloaded-leg experiment. If the separate
cold-mean contrast also yields no valid walking, the next feasibility screen
must isolate closed-loop DIAL locomotion: release Cartesian hand control,
remove box contact and contact-task scoring, and preserve leg authority,
locomotion weights, timing and search budget. This deliberately removes a
task block; it is not a one-parameter Walk-and-Push comparison or a replacement formal
environment. Require six seconds without a physical fall, at least one valid
forward landing per foot, 0.20 m body and 0.15 m landing-support progress.
Also retain balance/flight diagnostics. Passing this screen establishes only
unloaded locomotion feasibility, not force-safe walk-and-push or MGA validity.

That unloaded screen is now complete under
`results/_development/humanoid_mga_repair/unloaded_locomotion/`: seed 110,
300 steps, Nsample 64, Hsample 24, Hnode 6, four diffusion updates, and the
pre-registered current-joint inverse-mapped initialization. All preflight
contracts and the final source-hash check passed; the standard runner saved
the full trajectory in 1067.05 s. Hand and nonhand box force were exactly
zero throughout, and all states were finite. The six-second gate failed:
physical fall first occurred at step 120 (2.40 s), with only one left and zero
right valid forward landings. Before that fall, maximum body and landing-
support progress were only 0.10265 m and 0.13887 m, respectively.

The count is supported by physical events, not a missed-step hypothesis:
left unload at step 23, 2.93 cm clearance at 25, and loaded recontact at 43
advanced 35.88 cm; right unload at 47, 2.16 cm clearance at 51, and loaded
recontact at 59 instead moved backward 10.84 cm. Replaying all 119 pre-fall
contact-history transitions gives no mismatch. The first balance warning
occurs at 66 and the first full-flight interval at 69--79. Lateral motion is
a concrete remaining concern: robot COM y grows from 0.085 m at step 43 to
0.391 m at 60 and 3.018 m at 119; root lateral velocity is already 0.79 m/s
at the first valid left landing. The current locomotion reward penalizes
forward velocity only, whereas vendored DIAL penalizes both body-frame x
and y velocity. Removing contact therefore does not by itself fix walking,
but this result does not establish that eleven-joint planning is infeasible.

Pre-register one further unloaded **locomotion-objective-block** contrast,
not a reward sweep or a formal Walk-and-Push result. Preserve the entire previous
screen's physics, hand release, 23D action/11 planned joints, initialization,
budget, alive/balance/force penalties, terminal rules and physical gate. Only
replace its gait, velocity, upright, height, heading and angular-rate terms
with the vendored block, and add the vendored normalized torque-effort term:
gait weight 5; body-frame xy target (min(0.8 t / 2, 0.8), 0), using the
pre-transition physical clock; pelvis upright weight 0.01; torso height
target 1.2 m with weight 0.5; wrapped torso yaw target zero with weight 0.1;
and normalized actuator torque squared with weight 0.01. This explicitly
changes the forward reference from constant 0.25 m/s, not merely adding a
lateral penalty. Vendored code multiplies an already-radian angular velocity
by pi/180 before its yaw-only cost. For numerical reproducibility, express
that term directly in SI units with yaw-rate coefficient (pi/180)^2, about
0.000304617; do not describe the original multiplication as a valid unit
conversion or promote this diagnostic weight without validation.

Implement this contrast only as an in-memory reward-delta hook around the
existing task transition. Preserve every non-reward state field and success
padding. Torque effort uses the actual control computed at the start of the
20 ms interval, matching the vendor scoring point; the retained 4 ms x 5
feedback physics means it is not a substep-average effort or an exact vendor
environment reproduction. Other retained differences include gravity-
compensated posture-controlled arms, eleven rather than nineteen planned
joints, and clipping dense spline commands to the sampling bounds rather
than allowing vendor spline overshoot up to physical joint limits. Its
causal effect has not been measured; do not combine its repair with this
objective contrast. Before physics,
verify target values at t=0,1,2,3,
positive/negative lateral symmetry, body-frame rotation direction, actuator
normalization, semantic-hand-action inactivity, and unchanged transition
fields. The prediction is that a mismatched locomotion objective may permit
bilateral forward support without lateral runaway; merely surviving longer
does not pass. A failed block contrast establishes only that this objective
change is insufficient. No second case or canonical promotion is automatic.

The vendor-objective screen completed and saved in 1034.92 s (17.25 min),
with all states finite, exactly zero hand/nonhand box force and unchanged
source hashes. It fails the six-second gate: first fall is 279 / 5.58 s.
Before that fall, body progress reaches 4.02648 m and landing-support
progress 2.93330 m, with valid counts [2,1]. There are 22 single-foot real
recontacts, not merely three contacts; only right step 72 and left steps
166/230 satisfy the entire forward-step contract. All 278 pre-fall contact
history transitions replay without mismatch. Full-episode final/max body
4.48992 m and support final 3.83818 m include post-fall motion and cannot
qualify the gate.

The original lateral runaway is reduced: pre-fall COM y remains between
-0.20133 and +0.21277 m, versus the old +3.018 m by step119. However, the
unchanged static balance proxy flags 99/278 pre-fall transitions (35.61%;
first at59), and the eventual failure is a forward/left flip (upright
-0.03954 at279, torso height0.70564 m above its0.588 m fall threshold).
At200 the COM is only7 mm ahead of the load-weighted foot sites; at225 it
is355 mm ahead of the sole loaded right site. Later forward placement is
insufficient, the torso sinks, and flight/recontact precedes the final flip.
These loaded-site offsets are geometric observations, not CoP estimates or
a dynamic certificate. This result demonstrates substantial locomotion,
not force-safe 0.5 m walk-and-push or permission to waive the six-second gate.

The saved physical history also exposes a deployment mismatch before that
late failure. Static balance violations at59--67,92--101,123--130 and154--159
occur under alternating sole left/right support and subsequently clear. Over
these four intervals torso upright remains at least0.98159 and height at
least0.97521 m, without force/nonhand/fall violations. At125 the pelvis is
0.06093 m from the actual supporting foot but0.31128 m from the two-foot
mean; the latter includes the swinging foot and yields a positive0.06128 m
balance residual. Using true robot COM with that same two-foot mean does not
remove the problem. Current full-MGA's task override would force emergency
at the next action after59, even if a refined proposal could recover. This
is a code-path implication, not an executed MGA counterfactual or proof that
the interval is dynamically safe. Do not simply disable the balance gate.
Compare measured supporting-foot geometry at the same unchanged radius,
retain missing-support events explicitly, then test ordinary continuation
and emergency from matched actual MJX states. Saved compact q/qd alone is
not a full-state identity guarantee or a substitute for substep risk traces.

The same-radius supporting-set audit does not justify a direct gate swap.
Keep the pelvis reference and0.25 m radius; measure distance to the sole
loaded foot or nearest point on the two-loaded-site segment, and classify
zero positive foot load separately. The four early intervals above change
from33 distance violations to5, but before213 the alternative also introduces
16 previously unflagged distance violations and17 previously unflagged
unsupported states. Counting unsupported as non-certified gives38 early
triggers versus33 originally, starting at19 rather than59. Across300 steps,
the alternative has73 distance violations and46 unsupported states, versus
121 old mean-based violations. It flags81/88 states from213 onward, including
all22 physically fallen states, but not213--219. Thus actual support geometry
is more meaningful than the two-site mean, yet still cannot identify dynamic
recoverability or be promoted merely because selected early flags disappear.

The next isolated `unloaded_vendor_dense_chart` contrast retains this entire
objective block and every physical/arm/search/terminal setting. Only the
instance-owned dense-to-joint map changes to the vendor affine mapping
followed by physical-joint clipping; node sampling remains clipped. It is
not a global controller edit or a canonical promotion. In-range targets and
torques, all arm outputs, and the inverse-hold initialization must match;
out-of-range targets must match the physical-bound affine formula.

That dense-chart contrast has now completed all300 steps and standard
artifacts in1099.81 s, exit0, with two successful source-hash checks. It is
worse than the previous sampling-bound chart: fall149/2.98 s rather than
279/5.58 s; valid forward landings[1,1] rather than[2,1]. Before falling,
body/support progress are2.0971/1.3660 m; full-episode final2.8946/2.4505 m
include post-fall motion. Hand/nonhand box force remain exactly zero and
all states are finite. Physical event replay has zero mismatch. The fall
is a forward/left flip (up=-0.0662, height0.6936 m), preceded by torso descent
and the COM advancing beyond the loaded foot. Both cases' executed dense
leg actions remain within1+1e-6, so the demonstrated change is chiefly in
future candidate prediction and subsequent selected plans, not an observed
executed clipping impulse. Do not promote the physical-bound map simply
because it is more faithful to the vendor. Retain the complete failed case.

The next task-speed check, if authorized after this contrast is fully saved,
must branch from the vendor-objective screen with the original sampling-bound
dense map, not silently combine both changes. Change only the forward target
from0.8 to the task's0.25 m/s; preserve the two-second ramp, body-frame lateral
penalty, all other objective/physical/search settings and the six-second gate.
This tests whether the vendored continuous-pushing speed is appropriate for
the task controller's eleven-joint authority. It does not add a stopping law,
restore loaded hand contact or establish a force-safe push result. No formal
configuration is changed by an instance-owned diagnostic objective hook.

The task-speed screen subsequently passed the pre-registered unloaded gate
on development seed110. All300 steps/6 s completed without a fall; valid
forward landings are left103/118/229 and right74/136/173, with zero history-
replay mismatch. Final body/support progress are1.191789/1.065436 m, also the
episode maxima. Hand/nonhand box force are exactly zero and every state is
finite. Minimum torso upright is0.943439 and height0.799072 m; final upright
is0.998672 and height1.005128 m. The final50-step body-frame forward velocity
reconstructed from q/qd averages0.25634 m/s, consistent with the0.25 target.
Standard outputs, exit0 and both source-hash checks passed in1012.71 s.
The archived cross-case source difference is only the independently checked
Unjamming span edit in `box_push_brax.py`; do not claim identical source files across
the two screens. This is the first **unloaded DIAL feasibility pass**, not a
loaded Walk-and-Push task success, force-safe success or validation of full-MGA.

Static balance still requires scrutiny: the old two-foot-mean proxy is
positive on11/300 states, only159--163 and278--283, with maximum0.032049 m.
All eleven occur under right-foot support with the left foot lifted. At280
the right load is812.69 N, left clearance0.1673 m, upright0.995835 and height
1.00837 m; actual COM relative to the supporting foot is only(0.03372,
0.05048) m, while the mean-based proxy is positive. There are also10 states
with zero positive foot load (longest interval0.08 s); the old proxy is
negative in all of them and they recover without falling. Neither dropping
the proxy nor declaring all unsupported instants safe is justified by this
single successful trajectory. Do not modify MGA's hard gate without a
matched-state/deployment check. Next integrate the validated locomotion
objective as a task-scoped opt-in, then assess the loaded force/velocity/stop
protocol and paired MGA behavior before canonical promotion. No arm/Peg
configuration, checkpoint or historical result is part of this change.

The vendored implementation remains the Walk-and-Push reference, not a substitute task
contract. `baselines/dial-mpc/dial_mpc/envs/unitree_h1_env.py`, in
`UnitreeH1PushCrateEnv.step`, defines the two-second velocity ramp, body-frame
xy velocity tracking, pelvis upright, wrapped torso yaw, torso-height and
normalized applied-torque penalties. Its example uses a continuous0.8 m/s
target and does not require our force-safe stopping and dwell at a line.
Integrate only the validated objective through an opt-in H1 task mode in the
existing environment; retain the current action chart, physics, force limits,
success gate and legacy defaults. Do not silently port its contact-array
indices or treat its angular-rate scaling as a valid SI conversion.

The opt-in `walk_objective_mode: dial` is now implemented in the existing
H1 environment, with explicit velocity-ramp, height-target and effort fields.
It applies equally to every algorithm. The legacy reward/control paths are
unchanged;91D observations retain their width, while the startup clock and
policy interface explicitly identify the new objective. Old equal-width
checkpoints cannot silently load under a different objective contract.
Eleven selected new/legacy contracts passed in46.06 s, including one real
reset and no physical rollout. The first test attempt exposed reuse of a
JIT-cached test fixture after mutating its static configuration; using an
independent legacy fixture fixed the test without another production change.
No canonical YAML or loaded-force protocol has been promoted. Environment
SHA256 is `9cd8d3845a7616496577c669d853b1ed55f6c6de0136ff659aaeb0c539d9f51b`.

The first source-integrated loaded check was launched on2026-09-09 at
22:52 UTC under `walk_and_push_loaded_task_speed_objective/`: DIAL, development seed110,
300 steps,64 samples,H24,node6,4+10 diffusion steps and temperature0.05. Only
eight locomotion-objective fields differ from the archived loaded Walk-and-Push config;
the8 kg box,8 N slide frictionloss,30 N target,60 N force limit,0.5 m line,
contact geometry, arm impedance, `ki_force=1`, stopping/dwell and real-landing
requirements are retained. The11 planned joints use the existing DIAL-derived
absolute-target chart; this direct mode has no additive Unjamming ankle-brace term.
Unlike the successful unloaded diagnostic, Cartesian hand contact is active
and arm posture gains remain the loaded task's12/1.5. Preserve these differences
when interpreting the result. Record physical substeps and endpoint metrics
separately, plus actual contact acquisition, foot landings, force taper and
stop events. This run is development-only, not a formal replacement.
The first attempt stopped before any physical step because the diagnostic
read mass/friction from the original native template instead of the modified
execution `env.sys`. Preserve its manifest under
`walk_and_push_loaded_task_speed_objective_preflight_failed/`. The corrected preflight
verifies8/8 on the actual MJX arrays and records the native template's30/12
separately; the same-parameter first physical attempt restarted at22:56 UTC.
After all preflights passed, its first plan compiled in approximately218 s.

The corrected run completed with exit0 in1183.78 s, with all source hashes,
1500 actual4 ms samples and standard-plugin metric recomputations passing.
Task success and safe success are both0, with no valid forward steps on either
foot. The box enters the12 cm taper at step36 with10.827 cm remaining and
velocity1.30470 m/s. It first reaches the goal band at step40, but still moves
at1.31439 m/s: the body has advanced only4.9 cm, support has moved backward,
and goal hold never begins. It exits the band at41 and the robot first falls
at63/1.26 s. No target lies beyond the finite side edge before these failures;
such targets appear only after the fall. Walk-and-Push selects only the rear face, so a
negative hand-force contribution retracts the hand rather than pulling a
non-adhered box back from overshoot.

The box finally stops at1.35907 m advance,85.907 cm beyond the line. Under
the aligned8 kg/8 N sliding approximation, the taper-entry velocity needs
about85.1 cm to stop (84.8 cm to reach0.08 m/s), not the remaining10.8 cm.
The observed0.2 s coast from step40 to50 loses0.20001 m/s, consistent with
that1 m/s^2 friction-only deceleration. This supports the stopping-distance
mismatch for this trajectory, not a proof that all action sequences fail:
the planner can request less than30 N, and box-progress reward may favor
the observed fast push. Keep task-wide force/physics unchanged while testing
that distinction. Copying the vendor locomotion objective alone is insufficient.

Physical hand peak is63.3507 N at step12/substep1 (0.224 s), versus only
35.5252 N at outer endpoints. Exactly1/1500 substeps exceeds60 N; nonhand
box force is zero. Full-run physical nCVaR95 is0.530008, but this includes
real post-fall zero-force time and must not be described as stable walking.
Retain the full primary contract and raw samples, with an explicitly separate
first-fall-prefix diagnostic: through step63 this has315 samples, nCVaR95
0.614858 (largest16 samples under the existing ceil-tail convention) and
1/315 force-limit violations. First physical contact occurs at0.176 s,
first balance violation at0.628 s, and first substep fall at1.252 s.
The foot-history audit exactly reproduces all300 saved counters. Both feet
unload at steps12--16, then the right and left feet recontact27.3 and37.0 cm
behind their initial placements at17 and19. A later attempted swing is
cancelled by flight at40. This is not a valid alternating forward walk.
Likewise, the final body/support displacements
are not valid locomotion achievements after the latched fall. No loaded Walk-and-Push
version or training checkpoint is promoted by this result.

The next isolated contrast is implemented as `walk_box_goal_mode: coast`,
defaulting to the exact existing `position` path. Only strict H1 joint-target
locomotion permits it. Replace the existing box position cost with the squared
error of `x + vx*abs(vx)/(2*a)`, where `a` is positive finite x-slide frictionloss
divided by mass from the actual task-modified MJX arrays. This is a conditional
one-dimensional stopping-location heuristic after immediate hand release,
not a barrier or a guarantee under continuing push. Keep all reward weights,
force targets/limits, control gains, physics and stopping/landing criteria.
The new objective is a task-owned extension to the vendor locomotion block;
the vendor's separate contact-count reward is not reproduced by our existing
hand-residual/nonhand-force terms. Likewise N64 is a development budget, not
the vendor example's N2048. Failure at N64 alone cannot falsify the vendor
loaded action space.

Thirteen selected coast/legacy contracts passed in56.98 s with no physical
rollout: actual mass/friction validation, signed stopping location, unchanged
non-reward transition fields/padding, zero-speed equality,91D width and
policy-interface incompatibility for old objective checkpoints. The coast-only
300-step DIAL check launched at23:52:43 UTC under `walk_and_push_loaded_coast_goal/`,
seed110,N64,H24,node6,4+10 diffusion steps,temperature0.05. Runtime preflight
confirms that the sole configuration difference from the previous loaded
check is `walk_box_goal_mode`; `ki_force=1` remains unchanged. Freeze source
through export, retain all real substeps and failures, and assess foot
landings, early support loss, box/body separation and stopping independently.
No physical outcome or canonical promotion is implied by contract-test success.

The coast-only experiment is now complete in1175.78 s (19.60 min), exit0,
with matching source/input hashes and all1500 actual substeps. It still fails:
task success and SSR are0. Valid forward landings improve from[0,0] to[2,1]
(right36,left78/109); at78 body/support advances are0.3038/0.1963 m, and the
robot remains upright through100. However, first physical fall is117/sub3,
2.332 s, caused by the height threshold rather than the earlier run's flipped
upright vector. The box has already missed the stopping goal. Its first short
stop near0.656 m advance is not its final position: post-fall collisions move
it to2.5656 m advance,2.0656 m past the line. Full-run force peak/nCVaR95/
violation rate are210.753 N/1.140155/2.4%, with a424.492 N nonhand peak.
This is early locomotion improvement, not whole-run safety improvement.

The explicitly separate pre-fall diagnosis has582 actual substeps strictly
before the first fall, hand peak67.2883 N,nCVaR95=0.650166 (largest30 samples)
and3/582 force violations. Including the complete outer117 gives585 samples
and3/585, with the same peak/tail mean. First predicted coasting-stop overshoot
occurs at outer28: box advance0.18227 m,velocity0.84032 m/s and stopping error
0.03534 m. The initial hand contact ends at34/sub1,0.664 s,with10.418 N normal
contact intensity; the corresponding outer34 endpoint has advance0.28688 m,
velocity0.86117 m/s and stopping error0.15769 m. Do not mistake that20 ms
endpoint pose/velocity for the same4 ms event state or normal intensity for
world-x force. Hand contact resumes only at111/sub4,2.216 s,with63.596 N.
Torso height was already falling between100 and110, so later recontact alone
does not explain the collapse. Nonhand contact first appears after the fall,
at147/sub3. Preserve the full primary statistics and these distinct prefixes.

Pre-register one further same-source objective contrast: coast plus only
`w_prog: 8 -> 0`, retaining `w_box=5` and every force/physics/control/termination
setting. Between27 and34, the retained progress term gains about0.969 whereas
the additional stopping-error penalty is about0.124. This local scale check
does not control other score terms or prove a candidate-selection cause;
also, progress already saturates at the actual line and does not pay for
extra displacement beyond it. Removing that term tests whether it encourages
excessive line-before-stop acceleration. The new run launched at00:14:55 UTC
on2026-09-10 under `walk_and_push_loaded_coast_no_progress/`, seed110,300 steps,N64 and
the identical coast source hashes. It is a development contrast, not a new
formal YAML or permission to mix in UNLOAD, gain or sample-budget changes.

The no-progress contrast is complete in1370.61 s (22.84 min), exit0, with
matching source/input hashes through standard export. All1500 physical samples
are retained, coverage is1 and no success padding occurs. Task success/SSR
remain0. Valid left/right landings are3/1 (right60,left70/103/123), verified
against every saved foot-load/clearance event. First goal-band entry is55 at
0.440791 m/s, and the first stationary box advance is0.578609 m,7.86 cm past
the line; the required hold never starts. First physical fall is140/sub1 at
2.784 s, not the20 ms endpoint time2.8 s. The final box advance0.4620 m
occurs after falling and reverse contact motion, not a successful recovery.

Full-run hand peak/nCVaR95/violation rate are81.0578 N/0.716013/7 of1500;
the peak is post-fall at181/sub3 (3.612 s), and nonhand force remains0.
Strictly before the first fall,695 physical samples yield63.1575 N peak,
nCVaR95=0.586884 (largest35 samples) and2 violations. Including all of
outer140 gives700 samples with the same peak/tail mean and2 violations.
Initial contact ends at43/sub5 (0.860 s),3.9041 N, while the benchmark
reference is still30 N. All hand samples are0 from44 until renewed contact
at127/sub1 (2.524 s),63.1575 N despite a0 N reference. Lateral drift and
height loss begin before that recontact: torso height falls from0.951 m at123
to0.919 m at126, when COM y is already0.4037 m. Maximum pre-fall absolute
COM y is0.5964 m versus0.1913 m in the preceding coast run. Removing progress
reduces early stopping overshoot and delays falling, but does not establish
overall locomotion or safety improvement. No third reward sweep is launched.

Recheck the vendored DIAL coordinate and timing semantics before further Walk-and-Push
changes. The current opt-in objective matches its torso body-frame xy velocity,
wrapped torso yaw, pelvis-up vector,1.2 m torso-height target and slow-walk
foot-height phase. These checks do not make the complete environments equal:
the vendor controls19 joint targets, uses2048 samples and a0.8 m/s continuous
push command; this development task uses semantic Cartesian force-controlled
hands plus11 joint targets,64 samples and0.25 m/s, with additional stopping,
force and valid-locomotion conditions. Do not call a failed small-budget
force-control run a failed reproduction of the original benchmark.

The next read-only audit identifies a concrete startup-clock mismatch. The
benchmark reference waits for the0.30 s approach and then ramps over0.30 s;
actual NORMAL feed-forward instead ramps from the latched contact time. All
three recent loaded cases latch at outer1 (0.02 s), before measured contact,
because the15 mm reset gap is within the existing20 mm acquisition band.
Consequently, a zero benchmark reference does not imply zero commanded force.
The shared robot home pose does not imply the same contact initialization:
the vendor's box rear face is at x0.40 m and the hand front at x0.3315 m,
giving68.5 mm clearance. This task repositions the box to15 mm clearance;
the vendor has no corresponding contact-acquisition latch or automatic force
ramp. Preserve this distinction without simultaneously changing the reset.
Using saved actions, pre-transition contact time/integral and all five measured
forces reproduces each of the first15 outer-step integral endpoints exactly
in float32. These steps are before the distance taper, where the execution
reference is constant within each outer interval.

At the early physical peaks, the reconstructed command / ramped reference /
pre-substep integral / effective feed-forward are respectively
47.8699 /31.9133 /30 /60 N (loaded-position,12/sub1,63.3507 N measured),
52.5178 /38.5130 /30 /60 N (coast,13/sub4,67.2883 N measured), and
46.0508 /33.7706 /21.3912 /56.4740 N (no-progress,13/sub2,61.7854 N measured).
All three benchmark references are still zero. This corrects the earlier
interpretation of an exclusively position-impedance approach impact: early
feed-forward and integral are also active. It reconstructs scalar commands,
not unrecorded Cartesian errors, per-hand wrenches or world-x forces; the
later state-dependent taper cannot be reconstructed at4 ms from outer poses.
First test a Walk-and-Push-only synchronized startup ramp with unchanged acquisition,
force target, physical limits, KI, action sequence and rewards. Do not change
the benchmark to excuse early loading or infer full walking from a replay.
The read-only evidence is saved in `walk_and_push_startup_force_clock_audit.json` below
the repair development root (SHA256
`df06c2e97b7bdaf95c225dcd5bbcdd22f510a32f4e2efd4ebecefac1556d459a`).
It includes all225 scalar substep reconstructions and original artifact hashes.

The next intervention is prepared, not applied during the current freeze:
an opt-in `walk_force_startup_mode: synchronized`, restricted to H1
`push_walk`/`joint_target`/`locomotion`. Actual force scale is the minimum of
the existing acquisition ramp and unchanged global benchmark ramp, not their
product. The policy's last memory coordinate and saved nominal execution
reference must use the same helper; the action-selected force and actual
measured force remain distinct quantities. Default legacy behavior is retained.
After the five KI checks and the fixed-candidate gate/bridge diagnostic finish,
run focused contracts, then replay the saved no-progress actions for60 outer
steps in each mode on one frozen source. The legacy branch must first match
the old q/qd/obs/reward/force/integral/foot trace; no additional physics replay
is permitted for metrics or parity. The budget is600 simulated4 ms substeps
in total. Retain every failure and actual sample. A positive matched-action
result only motivates a fresh closed-loop DIAL check; it does not establish
walking success or justify a formal freeze.

A separate source audit also limits direct asset reuse: both the vendor and
local scene specify box diagonal inertia0.1 kg m^2, but the vendor has only
one x-slide whereas this task has x/y slides and a yaw hinge. Non-unjam
suites, including Walk-and-Push, constrain y/yaw to +/-1e-6 with soft joint limits;
Walk-and-Push is not a freely moving planar-yaw box, nor exactly the vendor's one-joint
system. Unjamming retains the free planar coordinates. Runtime mass/
size changes do not update this inertia. The local8 kg,half-width0.55 m cube
would have Izz=1.6133 kg m^2 *if uniform density were intended*, versus the
inherited0.1; a concentrated internal mass is a different possible model.
This is an unvalidated mass-distribution assumption, not proof of the impact
root cause or permission to change inertia to obtain a pass. Unjamming's yaw
frictionloss2 also differs from Walk-and-Push's8. Record actual runtime arrays in the
next Unjamming preflight and keep this audit separate from the current objective/
integral contrasts.

Before loaded validation, audit the force/velocity contract explicitly. With
the current development box mass8 kg and x-slide frictionloss8 N, an ideal
aligned steady push at0.25 m/s needs approximately8 N; following the two-second
linear ramp needs approximately9 N. Sustained actual x-force30 N instead
implies acceleration(30-8)/8=2.75 m/s^2 in this simplified sliding calculation.
The task's body-speed target is not box speed, and measured normal hand force
need not equal world-x force, so this is a consistency warning, not a proof
that loaded walking is impossible. Do not change mass, friction, force target
or safety thresholds merely to obtain a pass. Separate gait feasibility,
loaded contact tracking and approach/stop behavior, and freeze any justified
task-contract revision equally for all algorithms before formal evaluation.

A subsequent pure-sampling audit reproduced the actual unloaded cold first
batch through the DIAL-degenerate `MgaBackendJax` route, not an illustrative
random draw or a different backend. The archived sampler/bridge/plugin/profile/
spline hashes match; PRNGKey 1110 and its real split chain produce a dense
65-by-25-by-23 batch, independently reproduced with zero error. Initial
node pinning is exact; the saved first-action difference is only 5.96e-8.
All 64 random candidates (not the incumbent) exceed sampling bounds in at
least one leg coordinate: 989 of 17,600 leg/time entries (5.6193%), with
maximum absolute dense action 1.40339. Current sampling-bound clipping versus
vendor affine mapping followed by physical-bound clipping differs by up to
0.37905 rad in joint targets, and by 0.01759 rad RMS over all legs. All eleven
channels are affected. This establishes active prediction-authority mismatch
in the cold batch, not changed candidate rankings or fall causality. No
environment initialization or physical step was run; dummy reward weights
were discarded. Evidence is in
`unloaded_locomotion/cold_first_batch_joint_target_clipping_audit.json`.

Static vendor/current MJCF comparison found identical foot capsules/sites,
floor collision parameters, robot body/joint/inertial attributes, actuator
ranges, joint damping/armature, and default solver/iteration settings. H1
does not use the G1-specific explicit foot pairs. The non-rendering robot
geom difference is hand collision masks: current hands contact the box only
through explicit pairs, whereas vendor hands can also contact the ground.
This can matter during a fall or floor support, but no early hand-floor
contact evidence currently connects it to lateral drift. Do not change foot
friction or mass on the premise that these audited parameters differ.

The fall-objective audit also limits the interpretation of a future terminal
fix: the old 8/27 N traces earn respectively -83.87/-387.25 over the 25 steps
strictly after physical failure, despite +100 each from progress shaping.
Zeroing those tails would improve these bad sequences' scores. In contrast,
the old 8 N step-50 reward is +4.391, including +4.0 progress, while the body
and feet remain behind and failure is still 52 steps away. This supports the
short-horizon shaping contrast first; failure padding is a separate RL/MPC
protocol issue, not an established safety improvement.

This is only a low-level controller repair. Walk-and-Push's existing model-based balance/
tip certificate still uses the two-foot mean. Before promotion, actual support
geometry must be separated from the smooth control reference. A static pelvis
to two-site distance is not a dynamic balance certificate; a nominal flight
reference placed at the pelvis cannot certify support either. Actual
unload/lift, opposite-foot support and loaded recontact must underlie forward
step counting. Do not invent a flight grace period or treat a single frame of
zero foot load as proof of a fall. Candidate risk, emergency override, safety
index, saved metrics and reliability labels must share the same physical fall
and collision semantics. Any additional hard support predicate requires
validation against the saved physical contact signals before promotion.
The existing force/contact/balance/fall thresholds are now shared by candidate
risk, emergency override, safety index and persisted metrics. The fall test is
the actual environment threshold (0.6 times nominal torso height or flipped
upright axis), not the old metrics-only 0.5 m constant. Force-MAE labels use
the task's requested ramp/taper reference. Physical support event bookkeeping
is implemented, but this does not upgrade the retained static two-foot balance
proxy into a validated dynamic certificate. Learned reliability must be
recalibrated under an explicit matching label/horizon contract before full-MGA
is evaluated. Fixed-stance validation does not establish Walk-and-Push walking.

Historical Walk-and-Push isolated feasibility check (superseded by the direct-joint
closed-loop decision above): the opt-in support-phase/foot-level controller
passed its control contract but the Native MuJoCo screen fell backward at
1.46 s with zero valid forward landings and no hand contact. This screen
reuses the task controller and task-memory update, but Native physics is not
MJX validation. It establishes neither a working nominal gait nor an
impossibility result for nonzero leg residuals. Do not use success of this
zero-action CPG screen as a prerequisite for direct-joint MPC. The Walk-and-Push formal
promotion gate stays closed pending the actual closed-loop result.

At the 2026-08-27 execution gate, the H1 extension was marked complete. No formal
evaluation seed (10--19) was used for training, calibration, smoke execution,
or checkpoint selection. This authorizes starting P9; it does **not** claim a
formal performance advantage before the paired P9 matrix is complete.

| Frozen component | Non-formal protocol | SHA256 |
|---|---|---|
| Fixed-stance raw PPO | seed 0; 200k steps; Force Regulation/Fixed-Stance Push then Unjamming curriculum; final fixed-budget checkpoint | `1fe928bd143d7c03df8fcfe381aad4fc27a81ee50825187270f76643d394cf7d` |
| Walk-and-Push walk raw PPO | seed 0; 200k steps; final fixed-budget checkpoint | `9f5d25b9b078e7c3d87b5d2c72adf943e45682055a9ddc4c4a4983ef19a55d0e` |
| ATACOM Force Regulation/Fixed-Stance Push tangent PPO | seed 0; 200k steps; tangent width 5 | `242212e7604799a178414ff2028386757b4825a12ce017ce2f3323bc2ded81d0` |
| ATACOM Unjamming tangent PPO | seed 0; 200k steps; tangent width 1 | `887494c81832d024317b1960c95279cc6e8f456e2e7e491a45bd5faff0e01ec9` |
| ATACOM Walk-and-Push tangent PPO | seed 0; 200k steps; tangent width 22 | `66955be8bc550cbf249cb7b2cb1feeec4e3a10f0f73246f00ed822f4e05a1bd7` |
| H1 reliability | 600 seed-0 model-based development transitions; 600 disjoint seed-1 calibration transitions; one fixed 24-feature fit | `80303f6cb8d426244170d298111279f3bd75e6e274132c39a0ea41b2474e299b` |

H1's balance/friction inequalities contain structurally zero instantaneous
action-Jacobian rows. The task therefore selects ATACOM's finite damped-QR
projection, while Surface and Peg retain their historical SVD path. Random
exploration remained finite for 20 steps in all three H1 action schemas, and
the dedicated ATACOM backend tests passed.

The frozen verification record is:

- 35 canonical YAMLs pass the configuration audit;
- the checkpoint/protocol/causal audit reports exactly
  `Surface=1560`, `Peg=300`, `H1=740`, total `2600`, with zero errors;
- all eight H1 algorithms execute a development step in Force Regulation/Fixed-Stance Push, Unjamming, and Walk-and-Push
  checkpoint/action schemas (24 cases total);
- the four aligned H1 causal ablations and the Force Regulation `no_stiffness` diagnostic
  execute their declared development gates;
- the Peg MGA one-step run persists prior, revalidation, and emergency
  diagnostics; all three tasks persist schema-v2 task signals;
- 56 focused unit/audit tests and 22 non-slow unified-runner/plugin tests pass.

### P9 — Run formal matrices in causal order

Run order is important: establish causal core comparisons before spending
compute on broad baselines.

#### P9.1 Surface scan

1. MGA, Model-based Only, standalone RL, and DIAL on all thirteen suites.
2. MPPI, PegasusFlow, ISSA, and ATACOM on all thirteen suites.
3. Run `no_rl_prior`, `no_learned_reliability`,
   `no_controllability_geometry`, and `no_retraction` on all thirteen suites.
4. Verify completeness before aggregation.

Count: `8 * 13 * 10 + 4 * 13 * 10 = 1560` runs.

#### P9.2 PegInsert

1. MGA, Model-based Only, standalone RL, and DIAL.
2. `no_rl_prior` and `no_learned_reliability` ablations.
3. MPPI, PegasusFlow, ISSA, and ATACOM.
4. Verify strict safe-success derivation and all violation/jam events.

All ten configurations run all three suites.  Count:
`8 * 3 * 10 + 2 * 3 * 10 = 300` runs.

#### P9.3 H1 push

1. After the P8 H1 extension passes, run MGA, Model-based Only,
   standalone RL, and DIAL on all six suites.
2. Run MPPI, PegasusFlow, ISSA, and ATACOM on all six suites.
3. Run `no_rl_prior`, `no_learned_reliability`,
   `no_controllability_geometry`, and `no_retraction` on all six suites; run
   `no_stiffness` on both Force Regulation force steps.
4. Interpret Force Regulation with force-step metrics rather than line-goal success, Unjamming as
   the primary whole-body geometry comparison, and Walk-and-Push according to its actual
   generalization outcome.

Count: `8 * 6 * 10 + 4 * 6 * 10 + 1 * 2 * 10 = 740` runs.

The complete frozen protocol is therefore `1560 + 300 + 740 = 2600` runs.
This count excludes development smokes, checkpoint training/calibration, and
any optional stress-test appendix.

**Formal-v3 seed override (2026-09-03).** The three task matrices now use
evaluation seeds 0--9, paired across methods. This intentionally overlaps the
frozen seed-0 policy training and, for Surface/H1 reliability, seed-1
calibration provenance. The task YAMLs record an explicit overlap waiver; the
audit reports the overlap as warnings rather than presenting the split as
disjoint. Frozen checkpoint provenance and hashes remain unchanged.

Each matrix uses seeds 0--9, paired across methods. A run is incomplete if any
expected seed is absent, failed, or uses a mismatched config/checkpoint.

### P10 — Statistics, figures, and report verification

Actions:

1. Aggregate mean, standard deviation, and paired 95% bootstrap confidence
   intervals.
2. Use paired tests across identical environment seeds.
3. For multiple training seeds, use hierarchical bootstrap over policy seed and
   environment seed.
4. Produce the task tables and figures listed in Section 9.
5. Select representative GIF seeds by the locked median rule.
6. Run the completeness/provenance verifier.
7. Export paper-ready PNG and vector PDF where appropriate.

Acceptance:

- Raw and strict-safe success are both visible for PegInsert.
- Pareto plots include methods that win individual scalar metrics.
- No best-seed-only quantitative table is used.
- Every plotted value can be traced to a per-seed `results.json`.
- Confidence intervals and sample counts are printed.

### P11 — Update the paper only after the result lock

Actions:

1. Write the experiment section from the verified standard results.
2. Use the exact method label implied by the component contract.
3. State limitations where MGA does not win a scalar metric or where an
   ablation does not validate the intended mechanism.
4. Update tables/figures through reproducible export scripts.
5. Tag the result-producing commit and record the result root/checkpoint hashes.

Acceptance:

- Every headline claim is supported by a table, ablation, or confidence
  interval.
- The abstract does not claim independent RL-prior benefit unless P8/P9
  validate it.
- H1 is not called MGA unless the formal component contract is active.

### P12 — Retire legacy harnesses and development configs

Cleanup status (2026-08-28): the user explicitly authorized early destructive
cleanup after the formal configs, training recipes, and config-contract tests
were migrated. This does not promote development results to formal evidence;
P9--P11 and the result lock remain required.

Completed cleanup contract:

1. Solver-local experiment orchestration and config discovery are removed.
2. Formal and training configuration are owned by the unified config trees.
3. Obsolete development configs are recoverable from Git history but no longer
   coexist with the formal configs.
4. Legacy result/checkpoint data remain untouched and continue to satisfy the
   frozen learned-component bindings.

Acceptance:

- All formal commands enter through `genedynamics.experiments.runner`.
- No paper artifact depends on the legacy harness.
- The rollback tag and legacy result inventory remain available.

---

## 12. Formal completion gates

The experiment package is paper-ready only when all gates below are true.

### Architecture gate

- Unified runner is the only formal execution entry.
- All algorithms are independent plugins.
- Contact env/model/execution ownership is explicit.
- Standard result schema is used.

### Surface gate

- All eight algorithms complete all thirteen suites on paired formal seeds.
- `convex` is no longer missing.
- Hybrid stiffness tests validate or falsify the stiffness mechanism.
- Claims are Pareto-based, not "wins every metric."

### Peg gate

- All eight algorithms complete ID/PoseOOD/SensingOOD.
- Strict safe success is reported alongside raw success.
- No-prior and no-reliability ablations isolate the claimed mechanisms.
- Emergency/revalidation behavior is measured, not only described.

### H1 gate

- All eight algorithms complete all six suites with the preregistered
  suite-aware checkpoint binding.
- Force Regulation uses force-step metrics.
- The five targeted ablations are complete on their declared suites.
- Unjamming validates or falsifies tangent/retraction geometry statistically.
- Walk-and-Push is described according to its actual comparative outcome.
- MGA, MBO, RL, ISSA, and ATACOM labels match their effective components
  and checkpoint hashes.

### Reproducibility gate

- Clean commit, dependency/image identity, resolved config, checkpoint hashes,
  seeds, and exact commands are recorded.
- Verifier reports no missing or mismatched runs.
- Figures and tables regenerate from per-seed standard outputs.

---

## 13. Immediate next action

### PegInsert final freeze

PegInsert now has one versionless paper protocol, `mga_peg_insert_core_only`, and
one canonical configuration/result tree under `configs/arm/peg_insert` and
`results/arm/peg_insert`. Pre-freeze runs remain historical evidence only and
must not be pooled with the final matrix. Pose-OOD and Sensing-OOD seeds 0--9
were inspected during development, so they are evaluation-on-reused-seeds
rather than an untouched holdout; state this limitation explicitly.

The final decision is core-only MGA. It retains additive RL proposals,
model-based scoring and rollout, manifold/tangent shaping, local residual
correction, execution-time revalidation, receding incumbent, and task-owned
emergency recovery. It does not load a learned reliability checkpoint. The
Walk-and-Push development candidate remains unpromoted because four completed paired
comparisons showed identical insertion/full-window safe success and mixed
tail-risk changes versus the core controller. Walk-and-Push artifacts remain available
for audit but are not formal dependencies or paper rows.

The formal learned-component lock therefore contains only the frozen raw PPO
and ATACOM tangent PPO checkpoints. `no_rl_prior` differs from MGA only by
`use_rl_prior`; the historical `no_learned_reliability` entry is a non-formal
alias of canonical MGA and is not run. The second formal ablation is frozen as
`no_retraction` (w/o LRC). `no_tangent` is not part of the PegInsert formal
protocol.

Execution uses only the existing paper script and unified runner:

```bash
scripts/paper/mga/run_peg_insert.sh
```

The script evaluates eight final configurations with seeds 0--9 and writes the
240-run canonical three-suite matrix. `--resume` may reuse only complete,
config-matching runs. Policy digests, reward, physics, budgets and suite
definitions are frozen before execution. Do not tune from partial final
results. The existing development directories and logs are retained only for
audit and are not formal paper inputs.

Results retain the usual project-relative layout below each development root.
Use `genedynamics.experiments.utils.metrics` and
`genedynamics.experiments.utils.vis` for reports and rendering. Report safe
success, raw success, insertion depth, violations, force/torque peaks, jam,
emergency frequency, model revalidation/rejections and RL adoption. A
nominal predicted improvement does not establish true closed-loop dominance;
reliability diagnostics are development-only and are not expected in the
core-only formal MGA results.
Keep scanning configs, checkpoints, reward and existing results unchanged.

### Original three-task formal entry points

Start P9 through `scripts/paper/mga/run_all.sh`, or run the three task scripts
in the P9.1--P9.3 causal order. The scripts call only the unified runner, use
the frozen formal seeds 0--9, preserve explicit algorithm order, and pass
`--resume`. Do not tune configs or select checkpoints after inspecting partial
formal results. Keep the legacy runner, old configs, result files, and rollback
point until P9--P11 and the reproducibility verifier pass.
