# Four-Task Comparison Protocol: Interfaces, Permissions, and Acceptance Design

This document records research design and a code audit dated 2026-10-14. New P3 implementation and experimental facts are recorded separately in the
[P3 execution record](cross-embodiment-p3-execution.md); qualification samples are not formal comparisons.
By research decision, old CPU artifacts are no longer retained. Small-sample success counts in older documents are not results from this machine
and cannot serve as formal statistical replicates. P0–P2 development evidence from the current machine remains in the corresponding
`docs/development/` and `results/_development/` locations. The paper draft defines research questions,
methods, experimental protocols, and finally selected evidence; it is not an index of development logs.

See the [comparison plan](cross-embodiment-baseline-comparison-plan.md) for baseline literature selection and implementation decisions. This document first fixes the shared
environments, data permissions, and comparison criteria. Training had not started when the initial version was published; separately recorded training pipelines and physics qualification now exist.
Following the user's preference, the main literature pool now prioritizes 2025/2026 leading-conference work, especially Karen Liu and Guanya Shi.
The shared interfaces in this protocol must not silently remove a paper's native embodiment, supervision, actions, or complete workflow.

## 1. Evidence Roles and Actual Status of the Four Tasks

| Environment identifier | Task and question to isolate | Embodiment progression | Existing implementation and current qualification |
| --- | --- | --- | --- |
| `obstacle_transfer` | Carry an orientation-dependent payload around 3D obstacles into a workcell; separate known morphology/geometry from unknown execution gain, delay, and payload response | Panda source, xArm7 main target, UR5e cross-DoF extension | P3 has a public interface, successful source trajectories within the default 10 s, and exact CPU replay. Adapted Human2Sim2Robot PPO was trained in a nominal xArm7 model and executed on the real target environment. Sufficiently trained comparisons remain pending |
| `guided_contact_alignment` | Contact, slide, and align along real 3D guide surfaces; test whether friction/compliance changes require new contact strategies when free-space tracking is similar | Panda → xArm7, then UR5e | P3 has successful execution along both routes and exact CPU replay under the default 20 s, 25 N total-force/10 N lateral-force contract. New semantic GPU admission and recent trained baselines still need qualification |
| `keyed_insertion_recovery` | Matched-keyway insertion, jamming, and recovery; separate ordinary action compensation, contact control, task optimization, and distributional objectives | Panda → xArm7; report UR5e few-shot adaptation and frozen generalization separately | Retain P0–P2 findings on finite phase/affine models and CPU/MJX inequivalence. P3 adds successful source recovery with strict replay, public history/native-control interfaces, and three explicitly adapted learners. Recent strong-baseline performance still requires sufficient training |
| `mobile_keyed_insertion_recovery` | Retain the T3 connector, insertion criteria, and recovery semantics to test coupling between body support and contact reaction forces | Fixed/planar xArm7 diagnostics; real wheeled candidates; Go2 + Z1; G1 added separately if interfaces qualify | P3 newly executed insertion and two-mode jam recovery with real floating Go2 + Z1/OSQP stance WBC, matching strict CPU replay. Navigation/gaits, real wheels, new GPU topologies, and recent T4 baseline training still require qualification |

Relevant code paths relative to the repository root:

- `genedynamics/tasks/transfer/{runtime,specs,protocol}.py`: 13D skill, execution clocks, and task/mechanism contracts.
- `genedynamics/envs/domains/manipulation/transfer_workcell/queries/`: source-semantic queries and support markers.
- `genedynamics/tasks/transfer/contact.py`: contact profiles, termination criteria, and backend admission.
- `genedynamics/tasks/transfer/{behavior,contact_reference}.py`: explicit behavior phases, reference tables, and feedback decoders.
- `genedynamics/envs/domains/manipulation/transfer_workcell/`: workcells, numerical backends, and body compositions.
- `scripts/paper/cross_ebdm/`: unified paper entry points; task, physics, and algorithm implementations do not belong in scripts.
- `configs/arm/{obstacle_transfer,guided_contact_alignment,keyed_insertion_recovery}/` and `configs/quadruped/mobile_keyed_insertion_recovery/`: P3 qualification configurations for all four tasks exist; results continue to mirror domain/task/category/method/level/seed.

A runnable legacy MJX implementation does not qualify new spatial guides, TCP wrenches, or mobile compositions.
For new JAX contact semantics, `ContactRunner` currently admits only matched keyways with fixed Panda/xArm7;
Go2 + Z1 is explicitly limited to NumPy/CPU. Each new topology, contact profile, controller, and source
query requires its own acceptance check.

## 2. Freeze Shared Interfaces Before Choosing Algorithm Implementations

### 2.1 Observations, History, and Information Access

The main four-task comparison uses state/force observations, without adding vision as another variable for now. Specific fields,
units, frames, sampling rates, noise models, and missing-value masks enter configurations and schemas after development qualification.
The proposed public observations are:

| Field group | Permitted contents | Fairness requirement |
| --- | --- | --- |
| Task semantics | TCP/workpiece poses and twists relative to the workcell, goal pose, known workpiece/fixture geometry | All methods use identical measurable state; hidden friction, true effective gains, or future-feasibility labels cannot enter observations |
| Body and execution | Target encoder q/qdot, declared joint limits/morphology, previously issued commands, observable saturation and execution margins | Use masks or body-specific encoders for different DoFs; no direct correspondence between Panda and target joint vectors is required |
| Contact | Measured TCP wrench in the same frame convention, measurable permitted-contact/axial-support events, contact history | Separate commanded from measured force; internal ground-truth contact labels unavailable from declared sensors are evaluation-only |
| Time and behavior history | Elapsed time and equal-length observation/action histories; phases/events computed from public observations may be shared features | If Full uses hand-designed phase memory, dense baselines and controlled ablations receive the same event information, or it is removed for all |
| Mobile body | Base/torso pose and twist, support/slip observations, WBC projection/saturation/failure | All methods on a body use the same WBC and feedback; removing high-level body conditioning does not remove feedback from the common low-level WBC |

The source prior receives only task semantics declared when frozen and valid source virtual states. Target joints,
target-controller memory, and true body parameters cannot be inserted directly into Panda queries. The existing
`SourceQueryAdapter` supports only phase 0 for ObstacleTransfer; `ContactSourceQuery`
uses TCP pose/twist, normal force, and contact phase, and checks source-contact compatibility.
Valid source-query coverage requires separate acceptance; IK success does not establish that contact history can be reproduced.

Not every carry in the current code is automatically a “measurable observation”: MuJoCo internal warm starts, hidden
delay queues, and true model parameters are only for physical reproduction or audit. Maintain explicitly defined, separate structures
for public observations and simulator checkpoints.

In P3 `CpuTaskAdapter`, `previous_action` and `reference_position/rotation` come from publicly issued
bounded commands before hidden actuator delay; the plant's actual delayed action and integrated references are execution/audit-only.
Historical P2 `cpu_mechanism_validation.policy_step` still reads backend reference carry and cannot directly
join an equal-information main comparison. Before P4, migrate our policy/source queries into the same public adapter and rerun
development comparisons. Historical P2 results retain their original access conditions and do not establish a fair new ranking.

### 2.2 Actions, Low-Level Control, and Policy Capacity

The common skill has 13 dimensions: 6 for world spatial twist, 6 for Cartesian log-stiffness
ratios, and 1 for compressive normal force. The shared adapter handles coordinate transforms, velocity/acceleration/stiffness-rate limits,
force rate, projections, and low-level control. Common current clocks are physics
2 ms, control 20 ms, and skill horizon 10. These are implementation starting points; final values suitable for each of the four tasks
and different bodies still require individual qualification.

Mask the normal-contact-force channel uniformly when T1 does not need it. T4 does not add base velocity or gait labels as a
method-exclusive high-level action: the same WBC executes the common 13D skill. Body probes may use
a preregistered identification-specific interface. Any body-intent/stance-positioning module at deployment must be fixed and public
for all methods.

The current T3 `PhaseGaussianPrior` is a small model mapping 8 phase × mode classes to 5D references,
combined with a feedback decoder and finite affine-map search. It is not evidence from a complete dense-policy
study. Strong baselines should continuously output valid 13D actions/chunks and receive public history and
task goals; restricting all external algorithms to the existing 24 maps cannot justify a claim of sufficient comparison.

Capacity-matched task-only, removal of response matching, deterministic transport, and similar variants serve
attribution. External methods with different native structures may retain appropriate architectures, while disclosing parameter count, training data,
compute, and interface adaptations. All methods share collision, force, time, and body constraints;
relaxing the adapter or extending timeouts cannot create an advantage.

### 2.3 Separate Rewards, Task Costs, and Success Criteria

Freeze three objects instead of treating one training loss as performance:

1. **Observable common task reward/cost**: goal progress, orientation, time, measurable force,
   collision/execution risk, and necessary recovery events. Dense RL and model-based planners may both use it; freeze normalization,
   weights, terminal shaping, and accumulation only after development-set tuning.
2. **Execution success and safety criteria**: an independent evaluator recomputes them from actual trajectories for all
   methods. Stops, rejections, timeouts, source-support failures, and body failures remain in the denominator of all resets.
3. **Method-specific response/distributional objectives**: source anchoring, response discrepancy, mode
   coupling, etc. They may train Full, but do not automatically enter the shared definition of task success.

The current scripts' `.5/.25/.25` task cost, mode/time-weighted joint cost, and P1/P2
response signatures are development implementations; they cannot become the new four-task benchmark without an audit.
A claim of “improved response within a task-cost tolerance” requires preregistered task cost, tolerance, stopping cost,
and statistical comparison. The final force band in success conditions and the trajectory-wide peak-force limit
are different constraints and must be defined separately.

## 3. Numerical Conflicts Between Draft and Code: Resolve After Development Qualification

The table below records audit starting points and gaps, not new final thresholds. After viewing final tests, do not choose
a version favorable to a particular method. Ultimately generate the paper's protocol table from frozen configurations/schemas.

| Item | Draft starting point | Current code/development entry point | Issue to resolve |
| --- | --- | --- | --- |
| T1 timeout | 10 s | `ObstacleTransferConfig` defaults to 10 s; `validate_obstacle_mechanism.py --deadline` defaults to 30 s | Requalify frozen-source-prior support within the proposed shared deadline; do not mix historical 30 s results with a new 10 s table |
| T1 geometry/payload | L-shaped object bounding box 120 × 80 × 20 mm; mass 0.3–1.0 kg; workcell bounding box 0.8 × 0.6 × 0.6 m | `workcell.py` builds actual collision proxies/obstacles; default nominal payload in config is 0.5 kg | Export actual XML dimensions/masses, collision exclusions, and shared reachable regions individually; draft dimensions are not yet implemented asset specifications |
| T2 force/timeout | Resultant force 20 N, 10 s | `spatial_guide_config` inherits total-force limit 25 N, timeout 20 s, lateral limit 10 N | Choose an executable version consistently for all methods; no silent changes or aggregation across protocols |
| T2 initial states/shapes | Workpiece 80 × 50 × 20 mm, offsets ±10 mm/±10°, one-sided clearance 2–4 mm | Current spatial guide has a 12 mm tip, 16/18 mm channels, biaxial 0.1 slope; reset task noise 0.3 mm | This is a different scenario parameterization. Determine orientation changes, reset sampling, and truly recoverable regions rather than claiming complete implementation of draft disturbance ranges |
| T3 depth/hold | Nominal seating 15 mm, success depth ≥14 mm, 2°, hold 0.3 s | `research_contact_config` target displacement is 16 mm; evaluator checks Euclidean TCP goal distance <2 mm, 2°, force band, and 0.3 s hold | Current implementation is not an independent scalar “insertion depth ≥14 mm” criterion. Define and cross-audit socket frame, tool-tip depth, and complete orientation/envelope conditions |
| T3 workpiece/disturbances | Keyed object 30 × 18 × 12 mm, one-sided clearance 0.5–1 mm, offsets ±2 mm/±3° | Current matched keyway declares peg section 20 × 12 mm and key x extent 15 mm; `mating_contact_config` has clearance 0.8 mm and reset task noise 0.5 mm | Current composite tool cannot be equated directly with the draft object. Check actual reset-noise distributions and rotational support; numerical settings alone do not establish full disturbance coverage |
| T3 contact success | Draft emphasizes depth/orientation/hold, 25 N total force, 10 N lateral force, 15 s | Code additionally requires normal force within <2.5 N of desired 5 N; the matched profile also has a near-seat axial-support-loss criterion | Force holding and support failure must enter the common public evaluator; commanded force is not measured-force evidence |
| T4 body stability | Platform-specific stability, slip, and joint limits; 15 s operation deadline | Historical `validate_mobile_mechanism.py` uses development checks of min support ≥2, tilt ≤0.5 rad, QP residual ≤0.01; noisy task-space resets for Go2 are explicitly unsupported | Development thresholds do not automatically become new main-table specifications. Qualify standing, contact, slip definitions, QP-residual units/scale, and reset support |
| Probe budgets | Draft lists both 0/10/30/100 and 0/5/10/30 in different places | P0–P2 development calibration reused 5-probe units | Proposed new protocol first studies 0/5/10/30; decide whether to extend to 100 after development and before testing, keeping one version in text/config |

Retain P2's actual contact-peak definition: check both single-contact norms and net tool-force norms,
sampled at physics substeps. Freeze lateral-force definitions, frames, torque centers, and contact whitelists as well.
Control-period means cannot hide substep spikes. Halved-timestep/solver-convergence checks qualify the environment;
they do not offer extra target-tuning opportunities to any algorithm.

## 4. Body × Mechanism and Held-Out Splits

Start with a compact matrix that answers attribution questions, rather than immediately expanding the Cartesian product
of four tasks × all bodies × all parameters:

| Comparison axis | Required comparison | Supported interpretation |
| --- | --- | --- |
| Same body, different execution response | Nominal/shifted controller on Panda or xArm7, with known geometry unchanged | Calibration of unknown execution response rather than IK differences |
| Different bodies, responses matched as closely as possible | Panda → xArm7, with common task/tool and interface | Effects of known morphology/adapters; if Direct already solves it, this is interface evidence only |
| Simultaneous body and execution-response changes | xArm7/UR5e with controller deviation | Joint effect of cross-body adaptation and response calibration |
| Single-factor and combined contact changes | Controller first, then fixture friction/compliance, then preregistered combinations | Whether the contact model adds information beyond ordinary action compensation |
| Mobile-body coupling | Fix/release the same arm's base; arm-only/coupled calibration with the same WBC | Independent contribution of coupling between reaction forces and body response; locked base is diagnostic only |

Gain, delay, and friction values are not finalized as mechanism levels here. Draft gains 0.7/1/1.3, delays
0/20/60 ms, and fixture friction 0.15/0.35/0.6 are pilot starting points only. Wheel–ground/foot–ground
friction and body response need dedicated qualification; do not copy fixture ranges. Cross mechanism levels across bodies,
rather than permanently assigning the hardest mechanism to one robot.

Freeze data splits jointly by purpose and group:

- Source-prior data: successful Panda trajectories and genuine mode evidence. Split train/validation by episode
  group; normalization uses training only. Public response-development and validation
  data are likewise split by episode/mechanism group.
- Development set: environment qualification, rewards, model structure, hyperparameters, and probe allocation. Mechanisms/resets already
  used to debug P0–P2 remain development data and cannot be relabeled untouched final tests.
- Generalization sets: separately hold out initial states/goal poses, mechanism combinations, mechanism parameter ranges, and bodies; separate within-support
  from outside-support cases. Holding out only random seeds while reusing every mechanism is not mechanism OOD.
- Final target set: accessed only after all methods, models, stopping rules, mode classifiers, budget allocations, and hyperparameters
  are frozen. Its trajectories/failures cannot feed back into model selection followed by confirmation on the same set.

Report two UR5e protocols separately: freeze the prior/shared response/transport and fit only
context, or update transport within the same calibrated-twin budget. The latter is not frozen
generalization. If UR5e participates in public response training, label it a seen-body engineering condition,
separate from the unseen-body main protocol. T4 likewise discloses whether the target body participates in response pretraining.

## 5. Target Access, Simulation Budgets, and Statistics

### 5.1 Information-Access Levels

| Protocol | Target information/interaction | Purpose |
| --- | --- | --- |
| A-fixed: limited calibration, zero target-task training | Known morphology/control capability and the same charged short-probe buffer; no successful target demonstrations or complete target-task RL | Main four-task comparison. Offline updates query only the calibrated twin; ordinary residuals, dense policies, MPC/MPPI, and Full obey identical target access |
| A-active: native active-identification system | Method-specific short probes, with the same total count/duration/failure costs as A | Separate complete-system comparison; an identification module alone cannot represent the original method. Separate from shared-buffer mechanism ablations |
| B-log: complete self-collected target records | Charged complete action/observation trajectories, no target reward/expert demonstrations; extra state measurements listed separately | Native FADA/ASAP adaptation; methods in a group share recording access. Complete trajectories do not enter A merely because they lack rewards |
| B-online: target-task adaptation | Explicitly charged full-task rollouts and rewards, without extra demonstrations | Strong online adaptation and data-efficiency curves; complete task interaction cannot be hidden as short probes |
| C-supervised: extra-supervision reference | Successful target demonstrations, human corrections, aligned actions, or external pretraining, individually declared | Native supervised training, with target online interaction charged separately; cannot be mixed into A |
| D-oracle: sealed diagnosis | True target parameters, ground-truth-model planning, high-compute feasibility search | Explain task capability and mode support, without claiming a proven upper bound; provides no main-method training samples, mode masks, or tuning guidance |

Beyond probe counts, report maximum duration, actual duration, reset/warm-up time,
and total minutes. Interface adjustments, fitting, held-out physical validation, failures, and interrupted
attempts all count toward B; failed attempts do not earn free replacements. B=0 cannot read true target parameters.
Methods such as FADA requiring approximately two minutes of action/observation records need data points at the native
duration in addition to short-probe subprotocols. Separately declare whether data includes complete target-task execution;
not using rewards does not turn arbitrary full-task trajectories into short probes. Demonstrations and external pretraining remain declared under C-supervised.

Account for post-calibration GPU/CPU model-training calls separately from real target interaction. Model-validation
resets in a calibrated model are not additional target probes; conversely, validation/tuning on the CPU
target cannot be recorded as free model calls. P2's six reused calibration units do not become
more independent calibration replicates merely by adding resets.

Disclose simulation transitions, effective/padding steps, update counts, hyperparameter attempts,
parameter counts, compilation/training/reset times, and compute devices. Main comparisons show target-data budget curves,
then compare algorithms at matched simulation budgets and deployment latency. Online MPC/MPPI model calls and latency
count toward deployment cost; amortized policies cannot be compared only with unrestricted-compute planners while omitting costs.

### 5.2 Statistics and Full-Denominator Failures

Retain the original research target as a starting point for resource estimation: each formal cell should aim for at least 5 independent calibration/training
seeds and at least 50 frozen evaluation episodes per seed. Any reduction in cell count or sample size
must follow wall-clock/variance estimates and be fixed before viewing final tests; this document does not present these as
completed work or cost-free commitments.

Methods share reset/disturbance lists and physical initial states. Define random streams by global sample
identity so devices or batch reorderings do not change a world's random sequence. Resets for fitting, model
validation, and final target testing are independent. Cluster statistics by independent calibration/training seed and mechanism group,
retaining method pairing; 100 trajectories after one training run are not 100 training replicates.

Report overall, per-group, and worst-group safe overall success, task cost, stops/rejections/timeouts,
completion time, and confidence intervals. Failure/stopping is part of task performance. Speed and
force among successful samples may be conditional metrics with explicit denominators, but cannot replace overall metrics. In T4, stable non-insertion,
tipping, support loss, and WBC failure all count as overall failures. Source-support failures are not excluded.

The CPU target remains the current physics reference. Use consistent training backends across methods and report
calibrated-model-to-CPU-target error separately. Corresponding physics-replacement qualification is required only when GPU execution
replaces the target reference; CPU-confirmed research need not wait for equivalence of every long GPU trajectory.
Backend-induced differences in termination cannot support an equal-physics end-to-end speedup claim.

## 6. Fair Rules for Mode Semantics and Distributional Performance

Main-task success follows environmental goals and safety definitions. Without explicit user mode requirements, task-only
may legally select a more reliable successful mode; source-mode mismatch does not automatically become task failure.
If modes have semantic requests, such as a specified detour side or a forbidden recovery contact, preregister them as task
inputs and success conditions, and provide the same mode request to all methods.

Classifiers for actual T1 routes, T2 contact-guide/event sequences, and T3/T4 recovery events must be labeled from
real rollouts and frozen before final testing. The current T3
`classify_contact_modes` first requires withdrawal after loading, then separates two branches using a y-displacement threshold;
this coarse existing label cannot automatically be described as two complete strategy families of “search-and-advance” and “withdrawal recovery.”
Validate the identifiability of richer event semantics in successful source and target-development data before adding them.

Distributional metrics answer separate questions:

- Without mode requests: actual frequencies of successful behaviors, coverage of verified feasible modes, invalid/stopping
  probability mass, and empirical response distance; also report all resets and unknown-label mass.
- With mode requests: task success, actual behavior matching, and failures for each request; do not
  report mode matching only among successful samples.
- Response/law fidelity under equal task-quality constraints, alongside task-success–time–force
  tradeoffs. Entropy from random noise is not itself a contribution; deterministic transport
  may still produce multimodal behavior through a stochastic source prior.

Do not force preservation of source proportions when modes are infeasible. A sealed oracle labels a mode
verified feasible only after finding and successfully executing a behavior; failed search remains unverified, not proof of infeasibility. The oracle supplies no
training masks, demonstrations, or target ground truth. Cross-method response-distance comparisons must account for different rejection mass
and success denominators, not only each method's mean over accepted samples.

## 7. T4 Scope and Qualification Exit Criteria

The current planar composition in `contact_workcell.py` uses x/y/yaw driven joints, low-friction spherical
caster proxies, and mechanical constraints on height/roll/pitch. It isolates base degrees of freedom and common TCP
control, but cannot support conclusions about real tire friction, tipping stability, or general wheeled robots.

A real wheeled platform first requires selecting and freezing a chassis model, then validating mounting, mass/inertia, wheel–ground contact,
drive/velocity constraints, whole-body collision, and common force interfaces. In
`floating_workcell.py`, Go2 + Z1 has a real floating base, foot contacts, and a stance inverse-dynamics QP;
the host OSQP cost of 4 ms WBC must enter end-to-end latency. Current support is standing manipulation,
not gait generation. G1 still needs actual gripping/force interfaces, composite assets, and a controller;
it cannot enter result tables in advance.

The main T4 protocol retains the T3 connector, tool, force limits, timeout, and source prior. Initial states lie in the
shared support region of Panda and target, with contact history established by a public warm-up. Long-distance arrival or new workspace
outside source support is a separate extended-workspace setting: use the same navigation/stance-positioning module and retain
arrival failures and total duration. T4 manipulation modes inherit T3 semantics; support events are separate,
and gait changes do not count as manipulation-mode coverage.

First validate the same arm with fixed/mobile support, then real wheels, then one legged platform. A coupling contribution requires coupled
calibration to improve held-out response beyond arm-only/ordinary action calibration with the same WBC, and improve constrained task
or mode quality. If the public WBC explains all gains, accurately limit the contribution
to an interface extension. Claims across mobile support types require valid results on at least a wheeled and a legged platform;
humanoids are not a necessary commitment without interface validation.

## 8. Task Deliverables Across P3–P7

See the [new execution plan](cross-embodiment-p3-onward-plan.md) for phase numbering and complete criteria.
The old schedule compressing all work into small P3 steps is no longer used.

- **P3: native reproduction and comparison qualification.** First clarify four-task schemas, threshold conflicts, asset/source support,
  evaluators, and budget contracts. Qualify each baseline's native implementation and task adaptation separately;
  importable code or a loadable checkpoint is not complete reproduction.
- **P4: four-task development comparisons.** First integrate qualified recent methods in T1/T3; extend T2/T4 after their respective
  environment qualifications. Give ours and strong methods equal B-log recording access, plus shared-buffer controls. For C, use identical demonstrations
  or separate system-Pareto tables; assigning all strong competitors extra resources does not justify claiming overall superiority.
- **P5: method improvements.** Failure decomposition determines priorities for continuous feedback, models/backends, feasibility/stopping, calibration,
  and coupling. Interleave with P4; capacity-matched task-only and shared-WBC body ablations establish attribution.
- **P6: formal validation.** Once four-task qualification and development evidence are complete, freeze methods/protocols and generate new mechanisms,
  resets, and independent training/calibration replicates. Report full denominators, hierarchical intervals, resources, and per-task results together.
- **P7: paper.** Write contributions from final evidence. Native-reproduction records, engineering debugging, and unsuccessful versions remain in development
  directories. Task semantics, body scope, modes, and budgets follow this document rather than changing with final scores.

The T4 planar proxy is not a real wheeled platform, and standing Go2 + Z1 manipulation is not navigation/gait generation. Separate shared-WBC high-level
comparisons from native whole-body-control systems. Claims across mobile support types still require valid wheeled and legged evidence.
