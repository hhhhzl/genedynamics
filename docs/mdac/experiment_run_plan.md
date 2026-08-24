# MDAC Paper Experiment Integration and Run Plan

> Status: implementation plan, not an experiment result.
>
> Audit baseline: repository `5e10ca8` on 2026-08-24. The worktree was dirty at
> audit time, and several PegInsert/H1 files were still untracked. The scoped
> implementation snapshot containing this record tracks those files. Every
> formal run must use a clean checkout of that commit and record it in the
> standard experiment output.

## 1. Purpose

This document defines the complete implementation and execution order for the
three MDAC paper experiments:

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

The current
`genedynamics/solvers/single/mdac/run_experiment.py` is therefore a legacy
development harness. It remains available until equivalence tests pass, but it
is not a formal paper entrypoint and is not called by `scripts/paper/mdac`.

---

## 2. Immutable architectural rules

### 2.1 Ownership

The final ownership boundary is:

| Concern | Owner |
|---|---|
| MDAC, DIAL, MPPI, PegasusFlow, ISSA, ATACOM solver logic | `genedynamics/solvers/` |
| Robot models and task dynamics | `genedynamics/envs/` |
| Algorithm adapters used by the unified runner | `genedynamics/experiments/plugins/methods/` |
| Task adapters used by the unified runner | `genedynamics/experiments/plugins/environments/` |
| Metric extraction and shared metric math | `genedynamics/experiments/plugins/metrics/` and `genedynamics/evaluation/` |
| Per-seed visualization plugins | `genedynamics/experiments/plugins/visualizations/` |
| Cross-run aggregation, verification, and result rendering utilities | `genedynamics/experiments/utils/` |
| Paper entry scripts | `scripts/paper/mdac/` |

`scripts/paper/mdac` contains orchestration only. It must not implement a
solver, task reward, metric, statistical test, renderer, or result parser.
Runtime metric adapters remain under `genedynamics/experiments/plugins/metrics/`;
cross-run result/statistics utilities remain under
`genedynamics/experiments/utils/`. Task-script directories are not an alternate
home for either layer.

### 2.2 Algorithms are not MDAC variants

The formal algorithms are independent algorithm-level entries:

- DIAL
- MPPI
- PegasusFlow
- ISSA
- ATACOM
- standalone RL
- Model-based Only
- Full MDAC

Each algorithm has its own YAML and its own method plugin name. PegasusFlow,
ISSA, ATACOM, MPPI, and DIAL must not be encoded as cosmetic variants of the
Full-MDAC YAML. Shared adapter code is allowed internally, but the runner sees
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
| rigid cylinder | Full MDAC | 1.610 mm | 3.020 N | 0.0 | 0.370 mm |
| rigid cylinder | DIAL | 4.468 mm | 5.413 N | 0.0 | 0.502 mm |
| soft cylinder | Full MDAC | 2.267 mm | 4.930 N | 0.0 | 0.471 mm |
| soft cylinder | DIAL | 4.582 mm | 6.307 N | 0.0 | 0.534 mm |
| soft unseen | Full MDAC | 1.356 mm | 0.914 N | 0.0 | 0.413 mm |
| soft unseen | DIAL | 8.745 mm | 6.192 N | 0.0 | 1.031 mm |

These results support the following restricted claim:

> Realization-aware manifold/model-based control improves the
> precision--force--deformation tradeoff relative to unconstrained or weakly
> constrained diffusion/sampling baselines under the tested rigid, compliant,
> and unseen conditions.

They do not support either of the following claims:

- Full MDAC wins every scalar metric against every baseline.
- The RL prior is responsible for the scanning improvement.

Model-based Only and ATACOM are slightly better than Full MDAC on some scalar
metrics. For example, on soft unseen, their mean tracking errors are about
1.317 mm and 1.235 mm versus Full MDAC's 1.356 mm. The formal paper must report
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

| Suite | Full MDAC | DIAL | Model-based Only | ATACOM |
|---|---:|---:|---:|---:|
| ID | 80% | 40% | 0% | 0% |
| Pose OOD | 60% | 10% | 0% | 0% |
| Sensing OOD | 100% | 40% | 0% | 0% |

Full MDAC has zero force/torque violations and zero jams in these 30 episodes.
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

The audited H1 results compare DIAL with `model_based_mdac`; they are not a
comparison with the final RL-prior/reliability Full MDAC.

- P3 unjamming: model-based MDAC succeeds safely on 2/2 seeds; DIAL succeeds on
  0/2.
- P4 walk-and-push: both methods succeed safely on 2/2 seeds.
- P1 and P2 currently have only one completed seed per method.

The supported claim is:

> Task-owned contact geometry enables rear-contact unjamming/yaw correction,
> and the same model-based contact-control interface is executable in a
> coupled whole-body walk-and-push task.

The current evidence does not support:

- a claim that Full MDAC has been validated on H1;
- a claim that MDAC is statistically superior on P4;
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

- Full MDAC versus DIAL/MPPI/PegasusFlow/ISSA;
- Full MDAC versus Model-based Only and ATACOM to expose the actual Pareto
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

P1 is controller validation, P2 is nominal/OOD fixed-stance pushing, P3 is the
main geometry task, and P4 is a whole-body generalization extension. P4 is not
presented as a win unless the formal multi-seed results actually show one.

---

## 5. Target unified-runner architecture

### 5.1 Generic suite support

`ExperimentConfig` currently supports `obstacle_levels`, which is insufficient
for contact suites with different physics and task parameters. Add an optional,
generic `suites` field. A suite is a named evaluation condition, not an MDAC
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

Move the useful `base:` deep-merge behavior from the MDAC-private config loader
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

Full MDAC has one public algorithm name, but the resolved component contract is
recorded per task. The current internal methods are inconsistent across tasks
(`mdac_controllable`, `mdac_controllable_gate`, and `mdac`); this must be made
explicit and validated before any H1 result is labeled Full MDAC.

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
`if method == "mdac"` in metric computation.

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
│   │   ├── main/full_mdac.yaml
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
│   │       ├── no_stiffness.yaml
│   │       └── fixed_or_euclidean_stiffness.yaml
│   └── peg_insert/
│       ├── _base.yaml
│       ├── main/full_mdac.yaml
│       ├── baseline/<one-yaml-per-algorithm>
│       └── ablation/
│           ├── no_rl_prior.yaml
│           ├── no_learned_reliability.yaml
│           └── no_revalidation_emergency.yaml
└── humanoid/
    └── push_to_line/
        ├── _base.yaml
        ├── main/full_mdac.yaml
        ├── baseline/<one-yaml-per-algorithm>
        └── ablation/<only-causally-required-ablations>
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

All eight algorithms run the ten main suites. The three hybrid suites are a
targeted mechanism matrix: Full MDAC, Model-based Only, DIAL, ATACOM, and the
stiffness/geometry ablations. If compute permits, the remaining baselines may be
added, but their absence must be stated rather than filled with development
numbers.

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
p1_force_15n
p1_force_30n
p2_push_nominal
p2_push_ood
p3_unjam
p4_walk_push
```

P1 has a force-step success contract, not a line-reaching success contract. Its
metrics are rise time, settling time, overshoot, steady-state force error, force
violation, and balance margin. P2--P4 use task-specific safe push success.

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

- Full MDAC and standalone RL use the exact same RL checkpoint for a task.
- ATACOM uses its tangent-space checkpoint, trained with the same environment
  interaction budget and training-domain count as the corresponding learned
  methods.
- Policy-training seeds are distinct from evaluation seeds and recorded.
- The scanning ATACOM 200k versus Full/standalone-RL 2M mismatch must be removed
  before the paper comparison is valid.
- PegInsert currently uses a 200k learned-policy protocol. Retain it unless a
  preregistered budget study changes all learned methods together.
- H1 does not yet have a validated Full-MDAC training contract. It is not run as
  Full MDAC until that contract and checkpoint are established.

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

Do not create a second MDAC-only `metrics.json/manifest.json` formal schema.
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

P1 data/figures:

- 15 N and 30 N force-step response;
- rise/settling time;
- peak/overshoot/steady-state MAE;
- force violation and balance margin.

P2--P4 data/figures:

- safe push success;
- goal error/progress/completion time;
- yaw error and yaw recovery for P3;
- fall and balance violation;
- force peak/CVaR/impulse;
- friction-cone violation/slip;
- non-hand collision;
- wall contact/force where wall contact is task-defined;
- P3 top-down box/yaw/contact trajectory;
- P4 motion strip and executed GIF.

---

## 10. Paper-script contract

Create:

```text
scripts/paper/mdac/
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

1. Commit the scoped MDAC, task, robot-abstraction, metric, config, and test
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
6. Add smoke tests for all surface families, Peg suites, and H1 P1--P4 resets.

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

1. Add independent plugins for Full MDAC, Model-based Only, standalone RL,
   DIAL, MPPI, PegasusFlow, ISSA, and ATACOM.
2. Reuse existing solver factories and `run_receding`; do not duplicate solver
   update rules in the plugin.
3. Refactor only the minimum factory boundary needed to accept already-created
   model/execution environments.
4. Return the actual collected state/action trajectory and per-step infos.
5. Record the resolved MDAC component contract for every task.
6. Enforce the task-level fairness budget before execution.

Acceptance:

- Every algorithm is discoverable by its own method-plugin name.
- DIAL/MPPI/PegasusFlow/ISSA/ATACOM are not routed through an MDAC variant flag.
- One-step and short receding rollouts run for all three tasks.
- Full MDAC fails validation if a required prior/reliability/component is missing.
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
surface scan: rigid_cylinder, Full MDAC and DIAL
PegInsert: id_wide, Full MDAC and DIAL
H1: p3_unjam, model-based MDAC and DIAL
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
  `docs/mdac/freeze/p0_minimum_freeze.json`. This record is part of the scoped
  implementation snapshot; no dirty run is promoted to formal evidence.
- P1: base inheritance, deep merge, cycle detection, suite expansion, paired
  budget validation, and backward compatibility are covered by unit tests. All
  18 canonical algorithm configs resolve and dry-run, while the representative
  legacy `single_2d/mbd.yaml` protocol remains unchanged.
- P2: all 13 surface suites, 3 PegInsert suites, and 6 H1 suites create and
  reset through the unified environment registry; hidden execution/OOD
  environments preserve the robot observation/action interface.
- P3: eight algorithm-level method names dispatch their own registered
  controller. Full MDAC rejects a missing learned-component contract, and
  baselines cannot silently route through an MDAC flag variant.
- P4: contact metrics consume collected structured execution states. Standard
  seed results contain compact executed q/qd, controls, infos, nested task
  metrics, component provenance, and receding diagnostics; nested summaries
  retain count/mean/std/missing fields under strict JSON.
- P5: all six configured legacy/unified cases have identical resolved
  configs and bit-identical executed action arrays. PegInsert matches 21 DIAL
  and 65 Full-MDAC metric/diagnostic scalars; H1 matches 35 DIAL and 32
  model-based scalars. The surface legacy harness replayed actions rather than
  retaining executed states. Explicit replay of the unified DIAL and Full-MDAC
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
- No formal paper script references the legacy MDAC runner.
- No formal script uses semantic environment variables to alter the protocol.

### P7 — Run a two-seed integration matrix

Use development seeds, not formal evaluation seeds, for this gate.

Surface scan:

- Run every rigid/soft geometry for Full MDAC and DIAL.
- Specifically confirm that `convex` now completes.
- Run all three hybrid maps for Full MDAC and the key stiffness ablation.

PegInsert:

- Run ID/PoseOOD/SensingOOD for Full MDAC, DIAL, Model-based Only, and
  standalone RL.

H1:

- Run P1--P4 for DIAL and Model-based Only.
- Run Full MDAC only if its checkpoint/component contract is complete.

Acceptance:

- No suite crashes or produces missing/NaN headline metrics.
- Result layout, summaries, PNGs, and GIFs are generated automatically.
- The direction of the validated legacy results is reproduced.
- Convex/hybrid failures are resolved before formal seeds are touched.

### P8 — Lock learned checkpoints and causal ablations

Actions:

1. Fix per-task training domains, interaction budget, training seeds, model
   architecture, and checkpoint selection rule.
2. Retrain any unfair learned baseline, including the current scanning ATACOM
   budget mismatch.
3. Verify Full MDAC and standalone RL load the exact same task checkpoint.
4. Train/freeze learned reliability using training/development data only.
5. Freeze PegInsert `no_rl_prior` and `no_learned_reliability` configs before
   viewing their formal-seed results.
6. Establish the H1 Full-MDAC prior/reliability contract or explicitly classify
   H1 as model-based-only evidence.

Acceptance:

- Every checkpoint has a hash, protocol, training seed, step count, and domain
  list.
- No evaluation seed enters training, checkpoint selection, or calibration.
- Peg ablations differ from Full MDAC by exactly the named mechanism.
- The H1 method label accurately reflects the active components.

### P9 — Run formal matrices in causal order

Run order is important: establish causal core comparisons before spending
compute on broad baselines.

#### P9.1 Surface scan

1. Full MDAC, Model-based Only, standalone RL, and DIAL on the ten main suites.
2. MPPI, PegasusFlow, ISSA, and ATACOM on the same suites.
3. Full/Model-based/DIAL/ATACOM plus geometry/stiffness ablations on the three
   hybrid suites.
4. Verify completeness before aggregation.

#### P9.2 PegInsert

1. Full MDAC, Model-based Only, standalone RL, and DIAL.
2. `no_rl_prior` and `no_learned_reliability` ablations.
3. MPPI, PegasusFlow, ISSA, and ATACOM.
4. Verify strict safe-success derivation and all violation/jam events.

#### P9.3 H1 push

1. P1 force-step validation; do not rank algorithms by line-goal success.
2. DIAL, Model-based Only, standalone RL, and Full MDAC on P2--P4 if Full is
   validated.
3. Add ATACOM/ISSA only after their whole-body safety/task interfaces are valid.
4. Treat P3 as the primary geometry comparison and P4 as generalization unless
   the statistics demonstrate a stronger result.

Each matrix uses seeds 10--19, paired across methods. A run is incomplete if any
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
3. State limitations where Full MDAC does not win a scalar metric or where an
   ablation does not validate the intended mechanism.
4. Update tables/figures through reproducible export scripts.
5. Tag the result-producing commit and record the result root/checkpoint hashes.

Acceptance:

- Every headline claim is supported by a table, ablation, or confidence
  interval.
- The abstract does not claim independent RL-prior benefit unless P8/P9
  validate it.
- H1 is not called Full MDAC unless the formal component contract is active.

### P12 — Retire legacy harnesses and development configs

Only after P5--P11 pass:

1. Remove formal documentation references to
   `solvers/single/mdac/run_experiment.py`.
2. Keep a temporary deprecation wrapper if tests/users still depend on it.
3. Move genuinely reusable config-loading logic to the unified framework.
4. Delete obsolete development configs only after mapping every retained result
   to its generating config and obtaining explicit approval for destructive
   cleanup.
5. Do not delete legacy result data as part of code cleanup.

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

- All ten rigid/soft geometry suites complete on paired formal seeds.
- `convex` is no longer missing.
- Hybrid stiffness tests validate or falsify the stiffness mechanism.
- Claims are Pareto-based, not "wins every metric."

### Peg gate

- All eight algorithms complete ID/PoseOOD/SensingOOD.
- Strict safe success is reported alongside raw success.
- No-prior and no-reliability ablations isolate the claimed mechanisms.
- Emergency/revalidation behavior is measured, not only described.

### H1 gate

- P1 uses force-step metrics.
- P2--P4 have adequate paired seeds.
- P3 demonstrates the geometry contribution statistically.
- P4 is described according to its actual comparative outcome.
- Full MDAC label matches active components.

### Reproducibility gate

- Clean commit, dependency/image identity, resolved config, checkpoint hashes,
  seeds, and exact commands are recorded.
- Verifier reports no missing or mismatched runs.
- Figures and tables regenerate from per-seed standard outputs.

---

## 13. Immediate next action

The next implementation turn starts with P0 and P1 only:

1. make a scoped clean snapshot/rollback point;
2. add/test generic `base` and `suites` support in the unified framework;
3. prove all existing non-MDAC configs remain unchanged.

Do not create the full canonical config tree or launch new formal experiments
until P1 passes. Do not delete the legacy runner, configs, or result files during
P0--P5.
