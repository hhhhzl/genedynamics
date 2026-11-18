# ICDT Paper Narrative and Experiment Completion Roadmap

Migration handoff supplement (2026-11-18): see the
[device migration and research handoff](icdt-device-migration-handoff.md) for migration,
result validity, the latest nine-implemented/three-pending subtask state, and execution
order. This document retains research design and development history. “Formal release”
does not mean new formal results have been executed or independent mechanism benefits
validated. When leaving old artifacts behind, follow the reconstruction order in the
[cold-start handoff](icdt-cold-start/README.md). Historical summaries preserve research
knowledge; they are not completed formal experiments on the new device.

Updated: 2026-11-05 (America/New_York)

Target: RSS 2027

Manuscript: [`latex/latex_icdt`](../../latex/latex_icdt/)

Structural reference: [`latex/latex_mga`](../../latex/latex_mga/)


This document records the paper logic, current evidence boundaries, and implementation
order established in this discussion. The manuscript contains only methods, theory,
experimental protocols, and citable results. Diagnostics, phase states, invalid runs,
code migration, and outstanding work belong in development documents. Preserve existing
frozen results under their original protocols; narrative revisions do not rename or
reclassify them.

Decision in this revision: each of four task families has three **core** subtasks,
for twelve total. The four formerly optional extensions enter the formal experimental
design. This revision updates only the manuscript and development plan; the four new
items are not yet registered or implemented and cannot be described as qualified or
measured. Within each family, complexity progresses from basic events to compound
events and then new contact topology/disturbance recovery. Mechanism parameter changes
remain conditions within a subtask. This document also corrects the prior claim that
phase-gain selection validated ICDT mechanisms. Formal performance experiments are
not yet released.

## 1. Research Focus Established in This Revision

The paper now centers on this question rather than directly copying source actions
to the target:

> Source skills contain useful approach, contact, mode-selection, and recovery
> structure. Different bodies and controllers change the actual responses to those
> commands. With limited target interaction, how can we identify useful source
> responses, turn them into executable target behavior, and conditionally deviate
> when source structure conflicts with the target task?

This focus groups related work into three complementary directions:

- Human2Sim2Robot, Play2Perfect, TactAlign, and KITE show how object motion, interaction
  intent, touch, and precision-manipulation priors can supply cross-embodiment structure.
- ASID, AdaptSim, and FADA show that deployment differences require target identification,
  simulation adaptation, or execution-module adaptation.
- DIAL-MPC and ordinary response MPC use models for online reoptimization, with additional
  online computation.

ICDT connects source behavior structure to target execution response: target
interventions calibrate responses, a source-conditioned map produces target commands,
and selective response constraints determine when to preserve source behavior.
The task objective always remains active; source correspondence cannot define task success.

## 2. Six-Paragraph Introduction Logic

1. **Need for skill reuse.** Existing skills contain interaction structure beyond paths;
   collecting complete target demonstrations and retraining fully on target is expensive.
2. **What recent directions already solve.** Shared intent, object-motion priors, target
   identification, task-driven adaptation, and online planning have demonstrated value.
3. **Remaining gap.** Shared coordinates do not establish that the target can safely realize
   a source response. A more accurate model does not automatically decide whether to retain
   a recovery mode; pure task optimization may discard useful structure with limited data.
4. **Running example.** The same lateral correction can release a source insertion jam
   but create a target force peak. Correct transfer may require different amplitude,
   speed, timing, or even abandoning that mode.
5. **Method.** Limited interventions calibrate task-relevant responses; bounded residual
   transport changes actual requests. Task bands, history compatibility, model support,
   and execution phase jointly control source anchoring.
6. **Contributions and evidence.** Each contribution maps to an RQ, a set of main-table
   comparisons, and at most three principal ablations. Winning every cell is unnecessary;
   prespecified phenomena must be reproducible, attributable, and accompanied by counterexamples.

## 3. Three Contributions and Their Conditions

### C1: Behavior Transfer Defined Through Execution Mechanisms

**Claim.** Cross-embodiment transfer must distinguish the source command prior, target
skill map, target execution kernel, and final behavior distribution. Closer internal
commands or embeddings do not establish preservation of task events.

**Required evidence.**

- Transport must change actual requests after decoder, bounds, recovery, and selector.
- The same semantic task must yield correct event order across bodies, not just endpoints.
- Realize source modes where feasible on target; report deviations under infeasibility separately.
- Establish at least two distinct morphology cases among Panda→xArm7, Panda→UR5e,
  and fixed arm→Go2+Z1.

### C2: Intervention-Calibrated Selective Response Anchoring

**Claim.** Limited target interaction supplies task-relevant execution information.
Source response adds a preference only where task-compatible, history-matched,
model-supported, and relevant to the execution phase.

**Required evidence.**

- Compare full and no-calibration under the same total target-access budget in
  prespecified conditions.
- Isolate source-response gains with a control retaining the task band but removing response.
- Compare full and ungated response in source-compatible and source-incompatible conditions
  to expose selective anchoring's different roles.
- Improvements must appear in constrained task success or prespecified task events,
  not merely response MSE.

### C3: Conditional Event Analysis and Increasing-Complexity Validation

**Claim.** Task-event transfer depends on source margin, calibration error, anchoring
error, and unsupported mass. Test this conditional result across a complexity ladder
from constrained geometry to body-support coupling.

**Required evidence.**

- Twelve semantic subtasks span geometric timing, sustained contact, hybrid recovery,
  dynamic environments, and body stability.
- A six-row general table uses four shared metrics.
- Event figures explain the contact phase where success or failure occurs.
- Report matched-budget performance, end-to-end latency, and worst-group boundaries.

If clean ablations cannot show independent response-anchoring value, narrow C2.
If effective commands do not change, reduced internal loss cannot support C1.
If differences appear only in the stance system, the claim is a system-level
morphology case, not general transport gains.

## 4. Research Questions

| RQ | Question | Main contribution | Main tasks | Decisive comparisons |
|---|---|---|---|---|
| **RQ1** | Can limited target interaction identify task-relevant execution responses sufficient to improve transfer? | C1, C2 | `dual_route_carry` → `aperture_brake` → `sequential_aperture_reorientation`; `dual_channel_contact` as a contact reference | Full vs no-calibration; ASID, AdaptSim; budget curves and phase-completion events |
| **RQ2** | With sustained contact, phase transitions, and environment motion, when should source response be retained or relaxed? | C2, C3 | Guided contact: `dual_channel_contact` → `corner_guided_seating` → `edge_slide_pivot`; assembly: `jam_recovery` → `compliant_fixture_insertion` → `insert_then_lock` | Full vs no-response vs ungated; ASID, AdaptSim, response MPC; adapted DIAL-MPC; events and force peaks |
| **RQ3** | Can the same semantics transfer across arm DOFs and support morphologies with reasonable online cost? | C1, C3 | Panda→xArm7/UR5e; `stance_insertion` → `reposition_then_dock` → `disturbed_docking_recovery` | Direct source, ASID, AdaptSim, DIAL-MPC, response MPC; recovery, support, end-to-end latency |

RQ-level conclusions come from common patterns across designated tasks and matched
mechanism tests. Full need not dominate every seed, mechanism, and baseline in all
twelve subtasks. Before formal execution, freeze the minimum meaningful effect and
statistical rules using pilot variance; do not define “a majority” after seeing final results.

## 5. Four Environment Families and Twelve Core Subtasks

| Family | Basic event | Compound event | Third core subtask (pending implementation) |
|---|---|---|---|
| Constrained-space object transport | `dual_route_carry`: two-route obstacle avoidance while carrying an object, followed by pose holding | `aperture_brake`: complete object crossing through a physical aperture, short-distance braking, low-speed hold | `sequential_aperture_reorientation`: cross the first narrow opening, reorient in limited space, cross a differently oriented second opening, and brake |
| Guided-contact alignment | `dual_channel_contact`: two physical channels, loaded contact, mode-specific exits | `corner_guided_seating`: turn in an L-shaped guide, maintain load, reach a physical endstop | `edge_slide_pivot`: slide an asymmetric object along a loaded edge, establish physical pivot contact, and rotate into place |
| Connector assembly with recovery | `jam_recovery`: loaded jam, unload/retract, realign, insert and hold | `compliant_fixture_insertion`: passive single-axis fixture moves under contact | `insert_then_lock`: axial insertion, contact-mode transition, rotational locking, and pull-out resistance after releasing axial pressure |
| Body-support-coupled assembly | `stance_insertion`: the same recovery semantics under fixed-foot support | `reposition_then_dock`: body repositioning within support limits, settling, docking | `disturbed_docking_recovery`: physical disturbance during contact, preserve/recover contact and support stability, then hold docking after disturbance |

Eight current contracts reside in `protocol/subtasks/` under five existing config
families, with shared loader
[`subtask_contracts.py`](../../genedynamics/tasks/transfer/subtask_contracts.py).
Their state is:

- The first eight tasks are runnable in the current registry. The four third subtasks
  are designs defined here, with no configurations or registered entry points yet.
  Implement them in existing directories with pending status; do not launch them as
  a runnable suite before physics, source, and evaluators are ready. Design agreement
  does not freeze numerical geometry, public sources, or algorithm qualification.
- The 4/4 CPU source reports for `aperture_brake`, `corner_guided_seating`, and
  `reposition_then_dock` match current contract SHAs. The old 4/4 report for
  `compliant_fixture_insertion` has a different contract SHA and needs review and
  requalification. The other four existing subtasks still need fresh source qualification
  and evaluator review under the new semantic protocols.
- `reposition_then_dock` uses a task-provided body target, public fixed-foot WBC,
  a 12-control predock stability dwell, and recovery-before-docking event order.
  Old timed body drift cannot stand in for this task.

Mechanism axes remain train/development/heldout variations within each subtask;
two friction values are not different tasks. Three core subtasks require distinct
event relations, physical geometry, or body-coordination demands. Progression adds
behavioral requirements; it does not assume the third task has lower success for every method.

### Third-Subtask Designs and Implementation Boundaries

| Subtask | Required physical events | Prohibited shortcuts | Prerequisites and main research question |
|---|---|---|---|
| `sequential_aperture_reorientation` | Entire object clears frame one → reorients between frames → entirely clears frame two → brakes and holds | One fixed orientation through both frames; bypassing frames; advancing events before the tail clears; braking by frame collision | Two separate physical frames, relative orientation, and full-envelope evaluation per frame. Verify no common passable orientation but a collision-free rotation path between them. Study translation/angular response and phase-error accumulation; compare ASID, AdaptSim, MPC, and no-calibration |
| `edge_slide_pivot` | Establish loaded edge contact → slide to pivot → rotate under physical pivot constraint → seat and hold | A sphere merely changing yaw; lifting and rotating in air; reaching only the endpoint pose; using controller phase as contact evidence | The current spherical tip/fixed orientation is insufficient. Add an asymmetric object, physical pivot, contact-point trajectory, and angular response. Study response retention from translational to rotational contact; main ablations are no-response and unconditional |
| `insert_then_lock` | Finish axial insertion → enter a real twist-lock slot → rotate to lock → release axial pressure → resist a shared small pull force and hold | Adding only a rotation target; treating terminal IK pose as locking; relying on arm position holding, continuous pushing, or friction as pull-out resistance | Existing straight-slot shoulders prevent internal rotation; add bayonet/locking-slot geometry. A shared low-axial-stiffness or bounded-retraction test must show load transfer to the locking shoulder and an unlocked negative control. Do not claim retention after gripper release. Study axial/torsional phase incompatibility; torque is an RQ diagnostic, not a fifth main-table metric |
| `disturbed_docking_recovery` | Stabilize before contact → establish loaded contact → apply prespecified physical disturbance → preserve/recover safe contact and support → hold docking afterward | Ending before disturbance; teleporting the base; anticipating a hidden disturbance clock; foot changes/walking; checking only terminal success | Reuse fixed-foot WBC; add physical-substep external force, actual impulse records, and a controller state machine that can return to stabilization. Allow direct stabilization while maintaining contact; do not force retraction or prior contact loss. Trigger disturbance from physical task events with the same dose for every method in a condition; actors observe measurable states only. Use for morphology robustness, without requiring a main SOTA-performance claim |

Determine numerical geometry, thresholds, disturbance directions/impulses, and
train/development/heldout splits through source feasibility, joint body reachability,
and public-controller tests, then freeze them. Do not tune by target-method ranking.
Mechanism categories in existing contracts define design scope, not necessarily
implemented runtime parameter bindings.

The first edge-pivot version is limited to planar yaw rotation around a vertical
physical edge, with a common Cartesian orientation/impedance interface supplying
lateral edge pressure. The existing scalar normal-force channel cannot stand in for
arbitrary 3D wrench authority. Start insert/lock with a static fixture to isolate
locking topology, without compliant-fixture motion. The third task adds event relations;
it need not combine every mechanism from the first two.

The main table covers all twelve tasks. Main-text mechanism ablations avoid an
unmotivated full twelve-task factorial sweep. RQ1 emphasizes aperture and sequential
aperture; RQ2 emphasizes prespecified representative conditions in corner, edge pivot,
compliant fixture, and insert/lock; the support family mainly answers RQ3. The three
ablation names/objectives stay fixed. More tasks do not create new contribution or
ablation categories.

## 6. General Table and Targeted Studies

### Six-Row General Table

| Row | Role |
|---|---|
| ICDT | Full method |
| Task-only | Same-capacity internal system reference; not external SOTA |
| DIAL-MPC† | Online diffusion-style annealing/replanning control |
| Response MPC | Ordinary model-optimization control |
| AdaptSim† | Task-driven simulation adaptation |
| ASID† | Task learning after active identification |

† denotes a task adaptation in this repository. Each adaptation must state retained
author mechanisms, changed training chains, accessible data, and differences from
the native system. Public geometry/source priors remain qualification and diagnostic
references. FADA currently lacks nominal competence and moves to targeted study;
its low scores do not establish failure of the original method.

ACP and Human2Sim2Robot remain outside the table, as agreed. They enter resource-separated
RQ-specific comparisons only after their adapted policies pass source/nominal qualification.
Missing policies are not scored 0. Add Play2Perfect, KITE, TactAlign, or similar work only
when task interfaces and public implementations permit faithful reproduction; do not
remove a recent paper's core mechanisms simply to fill a row.

### Four General-Table Metrics

1. **Constrained success ↑**: complete the task while satisfying all collision, contact,
   force, support, and body conditions.
2. **Key-event completion ↑**: complete prespecified aperture crossing/braking, corner
   endstop, recovery/reinsertion, or fixture-settling events.
3. **Peak force ↓**: episode peak contact force at physical-substep resolution.
4. **Failure-penalized normalized completion time ↓**: completion time/time limit for
   successes; 1 for failures.

Physical-constraint violations, end-to-end p95 latency, mode completion, worst-group
success, model errors, and target-interaction budgets belong in RQ figures or appendices,
without expanding the general table. Weight the three subtasks within each family
equally; freeze any further four-family aggregation in the formal protocol. Support-
morphology demonstrations must not conceal regressions in the first three manipulation families.

## 7. Stories Supported by Current Results

The independent GPU0 formal v2 batch completed 36 method-by-task cells, 232 actor jobs,
and 11,600 physical outcomes. The table reports constrained success pooled over
source-like, nominal, delay/gain-shift, and payload-shift mechanisms on target bodies.
It is not the future equal-budget twelve-subtask general table.

| Method | Dual-route | Dual-channel | Jam recovery | Stance insertion | First-three-family mean |
|---|---:|---:|---:|---:|---:|
| **ICDT** | 100.00 | 98.75 | 42.19 | 86.88 | 80.31 |
| Task-only | 100.00 | 97.50 | 47.19 | 60.16 | 81.56 |
| FADA† | 0.00 | 0.31 | 0.00 | 0.00 | 0.10 |
| DIAL-MPC† | 100.00 | 0.00 | 36.25 | 83.75 | 45.42 |
| Response MPC | 100.00 | 99.69 | 48.44 | 70.78 | 82.71 |
| Public geometry/feedback | 100.00 | 97.50 | 50.00 | 87.50 | 82.50 |
| Public source prior | 0.00 | 0.00 | 62.50 | 87.50 | 20.83 |
| AdaptSim† | 98.44 | 99.06 | 62.81 | 86.72 | 86.77 |
| ASID† | 96.88 | 98.44 | 59.69 | 87.50 | 85.00 |

### Current Observations

1. **Simple tasks saturate and cannot establish transport gains.** Several methods reach
   100% on dual-route. ICDT exceeds task-only by 1.25 pp on dual-channel, with a 95%
   interval of $[-2.19,5.00]$ pp. This motivates aperture braking and corner contact,
   not a claim of a stable advantage from a small difference.
2. **Static recovery exposes current method weaknesses.** Jam recovery is 42.19% for
   ICDT versus 62.81% for AdaptSim: -20.63 pp, interval $[-33.13,-7.19]$ pp.
   All 185/320 ICDT failures are overforce. CPU paired auditing finds 158 shared
   ICDT/task-only overforce failures, 27 ICDT-only regressions, and 11 ICDT-only gains.
   Lateral force limits affect 160/185; `retreating=false` at termination in 185/185.
   All 27 unique regressions are mode 0 and had prior transport admission, but only
   8 still admitted transport at the terminating control. The problem therefore includes
   shared early-contact/reference-unloading recovery and delayed force consequences;
   tuning only the final tick's transport gate is insufficient. This makes event-sufficient
   response and contact-phase anchoring a method question, not a negative result to hide.
3. **The full system works on the supported morphology, but attribution is incomplete.**
   ICDT exceeds task-only by 26.72 pp on stance insertion, interval $[21.56,31.41]$ pp,
   and is close to AdaptSim/ASID/public references. Full and task-only currently differ
   in completion, selector, and projection. Report a system-level gain, not transport-only
   attribution.
4. **Four complex subtasks are runnable and have development paired pilots.**
   `aperture_brake` implements a four-sided physical opening, complete L-shaped object
   envelope, physical-substep frame-contact evidence, and short-distance braking/holding;
   fresh CPU source qualification is 4/4. `corner_guided_seating` also qualifies 4/4,
   with peak forces 10.57--10.61 N and lateral peaks 5.79--5.81 N. All four follow loaded
   contact, first segment, corner, second segment, physical endstop. Old-contract
   `compliant_fixture_insertion` qualification is also 4/4 (current binding requires
   revalidation): maximum fixture displacement 0.227--0.273 mm, peak speed
   1.92--2.56 mm/s, peak acceleration 0.637--0.666 m/s$^2$, no travel-end support,
   and completed relative insertion/tool-and-fixture settling. `reposition_then_dock` qualifies
   4/4 with target repositioning, predock stability, loaded recovery, and docking hold;
   maximum foot slip is 1.87 mm. These establish runnable environments, public sources,
   and evaluators. Development pilots with only one trajectory per mode do not establish
   formal overall rankings.
5. **Existing formal critical-contact results support only a local mechanism story.**
   Re-pairing 320 jam outcomes/method by actor instance, request, and reset hash gives
   ICDT 42.19% overall: below AdaptSim by 20.63 pp, ASID by 17.50 pp, response MPC by
   6.25 pp, and task-only by 5.00 pp; above DIAL-MPC by 5.94 pp. In the `latency_gain`
   mode 1 cell, ICDT reaches 40%, exceeding AdaptSim by 10 pp, ASID by 22.5 pp,
   response MPC by 15 pp, and matching task-only/DIAL-MPC. This motivates delayed-contact
   response research but cannot replace overall ranking. A fresh clean-actor paired
   pilot did not reproduce that performance gap; see item 7.
6. **Preliminary budget evidence is a negative model diagnostic.** Target-response held-out
   log MSE at 30/60/120 seconds is 0.007443/0.007861/0.008412; source/nominal held-out
   MSE is also nonmonotonic. There is no evidence that more target data consistently
   reduces error. In the fresh physical pilot, full (60 seconds) and no-calibration
   (0 seconds) both achieve 16/16 with zero violations. Full normalized completion time
   is 0.7185 versus 0.7572, but the mean difference -0.0387 is driven mainly by
   `payload_shift/mode_0` at -0.1783; two of four cells point the other way. This is
   a local timing signal, not a stable budget benefit.
7. **The fresh critical-contact pilot shows insufficient discrimination in the two current
   cell types.** Full, AdaptSim, ASID, response MPC, task-only, and three clean ablations
   all achieve 16/16 with zero violations on the same 16 fresh resets. All 128 physical
   evidence files pass independent re-audit. Full normalized time is 0.7185, compared
   with ASID 0.6984, AdaptSim 0.7400, and task-only 0.7213; full does not consistently
   lead. No-response-anchoring and unconditional-response-anchoring score 0.7169/0.7161,
   with no positive response-term or selective-gate increment. Response MPC's median
   episode-level actor p95 is 0.282 s versus full 0.079 s. This remains an actor-only
   development metric, not formal end-to-end latency.
8. **The old aperture pilot has been superseded by actual response-learning results.**
   On two fresh paired requests with servo gain 0.8, 1-control delay, and 0.65 kg payload,
   response-trained ICDT and AdaptSim achieve 2/2; task-only, ASID, and response MPC
   achieve 0/2. Failure-penalized normalized time is 0.9200 for ICDT and 0.9850 for
   AdaptSim. The checkpoint first fits source/target public responses, then computes
   the task-band/response objective. The old `semantic-phase` 0/2 and post-hoc fixed-gain
   results remain only as invalid development lineage. On `reposition_then_dock`,
   all five methods succeed on two paired requests: ICDT/task-only/AdaptSim/ASID/DIAL-MPC
   times are 0.7104/0.7576/0.7172/0.7332/0.7004. This provides a preliminary ICDT
   efficiency signal over task-only, but DIAL-MPC is slightly faster and there is only
   one reset/mode; it is not a formal ranking. Reset pairing and all 20 compressed
   evidence files from both pilots pass re-audit.
9. **Old phase-gain selection remains invalid; the new v2 learning chain repairs causal
   ordering.** Old aperture `semantic-phase` improved from 0/4 to 4/4, with normalized
   time 0.9204 at the 120-second label; both corner cells achieved 2/2; all moving-fixture
   variants succeeded, with selected-gain times 0.5093/0.5000. Retain these trajectories
   as development references. Code review found that the actor merely multiplied public
   planner outputs by fixed phase coefficients; response models were fitted after model
   selection and used only diagnostically. Aperture no-calibration/no-response were
   hard-coded to unit coefficients; full and unconditional shared action rules; corner/
   fixture no-response used manually larger coefficients. These treatments did not change
   the manuscript's response objective, and task-only used a different old actor. Their
   differences cannot be attributed to calibration, response anchoring, or selective
   gating, nor establish that gates have no benefit. Their budget curves are parameter-
   selection curves, not equal-budget ICDT evidence. New `semantic-response-transport-v2`
   checkpoints store source/target linear response weights and each candidate's task loss,
   calibrated/uncalibrated response loss, gate, and objective; loading recomputes selection
   term by term. The 0/30/60/120-second aperture success curve is 0/0/1/1; 120 seconds
   selects scale 1.25. Three clean ablations genuinely recompute their objectives, but
   all four treatments select 1.25 and achieve 4/4 evaluation requests. Thus the learning
   chain is operational, while this cell does not establish independent performance
   gains from calibration, response anchoring, or selective gating.
10. **The obstacle family's third task is implemented and released for formal comparison.**
    `sequential_aperture_reorientation` uses two orthogonal physical frames; geometry
    auditing finds no fixed yaw that passes both. Final source qualification is 4/4,
    with two four-bit frame masks recorded per physical substep. All five methods achieve
    2/2 in the two-mode pilot. ICDT/task-only/AdaptSim/ASID/response MPC normalized times
    are 0.8709/0.8923/0.8073/0.8918/0.8914. ICDT is faster than task-only/ASID/MPC,
    while AdaptSim is faster still. Dual-aperture transport is admitted only in the
    terminal phase after the entire object clears both frames: residuals during crossing
    or adjacent alignment cause envelope bypass. Retain those failures. Existing formal
    dual-route results, paired pilots for the two new cells, and the actual learning chain
    release the three-task family for 100 unopened formal episodes.

Audit references:
[`semantic_phase.py`](../../genedynamics/solvers/single/cross_ebdm/learning/semantic_phase.py),
[`train_aperture_brake_response_transport.py`](../../scripts/paper/cross_ebdm/train_aperture_brake_response_transport.py),
[`run_contact_complexity_paired_development.py`](../../scripts/paper/cross_ebdm/run_contact_complexity_paired_development.py).
This revision corrects documentation interpretation only, preserving original artifacts,
labels, and SHAs. Implementation repairs and renewed comparisons follow the order below.

### Claims Not Currently Supported

- ICDT cannot be claimed to outperform ASID or AdaptSim overall; its first-three-family
  mean remains lower.
- Old public phase-gain selection cannot be presented as response-aware transport gains.
  The obstacle formal cohort is released based on the v2 learning chain and same-reset
  comparisons, but clean ablations still choose identical actions, so independent
  calibration/anchoring/gating performance gains are not established.
- Current poor FADA adaptation scores do not establish failure of its original mechanism.
- Go2+Z1 results do not establish navigation or learned locomotion; this is fixed-foot stance.
- Panda→UR5e is not completed; the current formal fixed-arm target is xArm7.
- Equal-budget sample-efficiency or latency superiority is not established. Existing
  main-table budgets differ, and comparable six-method end-to-end p95 is unavailable.

## 8. Three Principal Ablations

Keep only these three as the principal main-text ablations:

| Name | Objective/model change | What must remain identical | Question |
|---|---|---|---|
| No calibration | Use nominal response model/context | Network, warm state, training steps, selector, execution interface; only data access differs | Do target interventions provide useful information? |
| No response anchoring | $C+\beta h+\Omega$ | Task band, model, data flow, network, optimizer, sample stream, execution | Does source response add independent value? |
| Unconditional anchoring | $C+\beta h+\lambda R+\Omega$ | Identical to full except removing gate selection | Does selectivity avoid source bias? |

Existing task-only contains only $C$, removing both response and task band. It remains
a system reference and cannot replace the no-response ablation. All three ablations
must share actual command authority, mode policy, checkpoint budget, online admission,
and recovery wrapper. If these differ, report a system comparison rather than
single-factor attribution.

## 9. Current Theory Gaps and Revision Directions

### 9.1 Theory to Retain

- The command law and target execution kernel jointly determine the behavior law.
- Reweighting cannot create commands outside source support; transport can move support.
- Task-relevant identifiability requires only that the probe null space not affect
  the task quantity.
- Conditional event bounds separate source failure, small margins, calibration/anchoring
  error, and unsupported mass.

### 9.2 Required Strengthening

1. **Response descriptors are not yet event-sufficient.** Current short-term increments
   and control-rate peaks do not encode recovery order, sustained contact, physical-
   substep impacts, fixture settling, or body support. Define shared event responses:
   event times/order, within-phase contact coverage, substep force risk, relative fixture
   motion, and body feasibility. Differentiable training proxies are acceptable;
   independent evaluators still determine formal success.
2. **Short-term bounds do not directly imply whole-task success.** Write finite-horizon
   composition by phase, estimating response error, margin, and support risk separately
   and making composition losses explicit. Start with a finite-phase union bound or
   stopping-event decomposition, without claiming uniform infinite-horizon guarantees.
3. **Ensemble spread is not an upper bound on calibration error.** Current dynamic-fixture
   development data show insufficient spread/error correlation. Treat spread as an
   empirical gate feature. Theoretical use requires independent holdout coverage
   calibration, or placing uncalibrated components in $p_{\rm bad}$.
4. **Self-collected logs do not guarantee identification of new contact branches.**
   Report task-relevant excitation, event coverage, and unvisited branches in both theory
   and experiments; aggregate MSE does not replace coverage.
5. **Conditional event bounds do not prove superiority to task-only.** ICDT and matched
   task-only have the same expressive capacity. Frame the advantage as source-informed
   bias under finite data/optimization, tested through ablations and budget curves.
   Do not claim a sample-complexity theorem without explicit new assumptions and proof.
6. **Separate training gates, checkpoint fallback, and online admission.** $G=0$ removes
   only that sample's response term; it does not imply runtime fallback to task-only.
   Theory notation and method diagrams must distinguish all three selections.
7. **Execution interface $\eta$ is the attribution boundary.** Completion, projection,
   selector, recovery, and WBC changes alter the actual trajectory law. Fix $\eta$ for
   component claims and use issued requests to show that treatments reach physical execution.

After revision, RQ2's positive/negative contact cells should directly correspond to
large-margin supported, small-margin, and unsupported conditions in the event bound,
not post-hoc explanations of gating based on wins and losses.

## 10. What Remains Before a Paper-Level Main Table

### Complete Task Validation

- `aperture_brake` has a four-sided physical opening, complete L-shaped object-envelope
  evaluator, physical-substep frame-contact evidence, public braking/holding source skill,
  4/4 CPU qualification, v2 response learning, and a 0/30/60/120-second budget curve.
  The five-method paired pilot passes. All three ablations execute valid objectives
  but select the same residual; independent mechanism gains need more discriminating cells.
- `sequential_aperture_reorientation` has orthogonal frames, a fixed-orientation
  infeasibility audit, 90-degree inter-frame reorientation, per-frame contact bitmasks,
  4/4 source qualification, and a five-method paired pilot. Its formal 50-episode
  protocol is frozen but unexecuted.
- `corner_guided_seating` has a physical corner guide, public source, loaded turning,
  physical endstop, 4/4 CPU qualification, and a two-cell/two-seed/eight-method paired pilot.
- `compliant_fixture_insertion` has public post-contact excitation, physical-substep
  evidence, travel-end rejection, relative-insertion and shared settling criteria, and
  development comparisons including DIAL-MPC. The old 4/4 source report's contract SHA
  differs from the current contract; review bindings and requalify.
- `reposition_then_dock` has target repositioning, a stability window, closed-loop docking,
  4/4 source qualification, and a five-method paired pilot. Add independent resets and
  support/mechanism shifts to test whether ICDT's time signal over task-only persists
  across seeds.
- Requalify the four existing tasks under new semantic contracts; do not relabel old results.
- Implement the remaining three third subtasks according to Section 5. Each needs real
  physics, a public source, an independent evaluator, anti-bypass negative controls,
  and source qualification. Keep unfinished tasks unimplemented; do not fill new columns
  with old-task results.

### Complete Method and Baseline Validation

- First release shared early-contact/reference-unloading recovery, then validate
  force-aware admission covering delayed force consequences. CPU diagnostics have
  ruled out transport alone as the cause of all 185 failures.
- Three clean ablations are frozen as no-calibration, no-response-anchoring, and
  unconditional-response-anchoring. Full plus three ablations completed eight
  mode-specific CPU preflights, GPU training, and selectors. Within each mode, batch-index
  streams and task-only initialization match exactly. The 128-episode paired task pilot
  completed but fully saturated, so it should not simply scale to formal size.
- Qualify FADA, ASID, AdaptSim, and DIAL-MPC individually on source/nominal, documenting
  native mechanisms and task-adaptation boundaries.
- Fix the formal table at six rows: ICDT, task-only, ASID, AdaptSim, response MPC,
  DIAL-MPC. Repair corner/DIAL observer-interface gaps before judging applicability.
  Unconnected methods are pending, not 0; engineering gaps do not directly imply
  theoretical inapplicability. Retain FADA/ACP/H2S2R only for qualified targeted studies.
- Qualified ACP/H2S2R targeted studies do not block the general table.

### Complete Statistics and Resource Accounting

- Freeze interaction-time grids, independent training-instance counts, per-instance
  condition counts, and minimum effects after pilots.
- Charge all target data, active probes, complete task rollouts, model selection, and failures.
- Prespecify train/development/heldout mechanism splits.
- Preserve paired initial conditions and resample hierarchically by training instance
  and physical initial state.
- Measure six-method end-to-end p50/p95, deadline misses, GPU time, and wall time consistently.
- Report by family/subtask so pooled means cannot hide zero success in a contact/support group.

### Complete Morphology and Visualization Evidence

- Qualify same-task reachability and source skills for Panda source, xArm7 target,
  and UR5e target.
- Generate source→target episode GIFs with matched initial conditions, semantics, and cameras.
- Use `stance_insertion` to demonstrate morphology feasibility and support constraints;
  headline performance still depends on the first three manipulation families and matched
  morphology comparisons.
- If adding hardware, provide at least one contact task's source/target transfer,
  forces, and full failure denominators. Without hardware, titles/claims remain
  cross-embodiment simulation, not sim-to-real conclusions.

## 11. Naming and Code-Migration Rules

New manuscript text, configurations, runner outputs, and analyses use semantic IDs only:

| Historical alias | New semantic ID |
|---|---|
| T1 | `dual_route_carry` |
| T2 | `dual_channel_contact` |
| T3 | `jam_recovery` |
| T4 | `stance_insertion` |

Migration rules:

1. Do not rename old results, SHAs, reports, or historical commands; they are immutable evidence.
2. Map old `T1`--`T4` to semantic IDs only at a legacy-reader boundary. New manifests,
   results, figures, CLI help, and LaTeX must not produce numeric aliases.
3. Keep `configs` organized by existing robot/task family, with subtask semantics in
   `protocol/subtasks/`. Do not create `results/<project>/<run_id>/`.
4. Keep `scripts/paper/cross_ebdm` as the paper entry. Gradually migrate active scripts
   such as `run_t2_*`/`train_t3_*` to general entries driven by configs and semantic IDs.
   Algorithms, task physics, evaluators, and statistics stay in existing package layers;
   paper scripts provide thin orchestration.
5. Keep algorithm internals in `genedynamics/solvers/single/<method>`; task-specific
   controllers/evaluators in `genedynamics/tasks/transfer`. No algorithm names in
   high-level `learning`.
6. Before each migration, use `rg` to classify active code, compatibility readers,
   and immutable evidence. Rename active code only. Never change historical result
   contents for cleanup.

## 12. Implementation Order and Release Criteria

### A. Freeze Semantics and Evidence

- The design is four families of three tasks; Section 5 defines twelve names/event
  chains. This revision adds no runtime or configuration files. Later add four pending
  contracts and shared-loader registrations in existing directories.
- Active runners stop treating numeric aliases as canonical identities.
- Eight runnable registrations do not imply complete qualification under the new protocol.
  Add four existing-task qualifications and compliant contract binding per Section 5;
  qualify each of the four new tasks after implementation.
- Current v2 results serve only as an evidence preview for four existing subtasks.

**Exit:** contract, legacy-read, and result-path tests pass; historical artifact hashes
remain unchanged.

### B. Static-Contact Attribution

- CPU paired diagnostics are complete. Next, release shared early-contact/reference-
  unloading recovery in paired tests and validate delayed-force-aware admission on
  heldout requests.
- Train a nominal-response artifact with matched architecture.
- Complete no-calibration, no-response-anchoring, and unconditional-response-anchoring
  under matched execution. New corner/moving-fixture phase-gain selection cannot
  replace these ablations; compare again using the genuine ICDT learning chain.
- Explain causes using contact phases, first impacts, issued requests, and substep peaks.

**Exit:** full and each ablation differ only in the declared factor; source/nominal
qualification passes; all failures are retained. If response adds no independent value,
narrow the contribution before scaling.

### C. Complete New Complexity Levels

- Aperture, corner, compliant fixture, and reposition/dock have physics, evaluators,
  source skills, and development pilots; qualification boundaries remain as in Section 5.
- Sequential aperture has physics, negative geometry audits, source, independent
  evaluation, paired pilots, and formal-protocol release. Edge pivot, insert/lock,
  and disturbed docking remain required third subtasks. Parallel implementation is
  allowed, without skipping geometric feasibility, negative controls, or independent evaluation.
- Per task: physics/evaluator first, then source skill, public feedback, learned method.
- Use small paired pilots to establish discrimination and absence of evaluator bypass.

**Exit:** all twelve tasks pass physics/source/evaluator qualification. The four third
subtasks move from pending to runnable, with anti-bypass negative controls and at least
one successful non-oracle public reference.

### D. Baseline Qualification and Twelve-Task Pilots

- Qualify each of the six formal methods on source/nominal for every task.
- Provide reasonable native training budgets and record learning curves/failure causes.
- Decide targeted ACP/H2S2R studies independently, without delaying the main table.

**Exit:** every general-table cell has a qualified executable policy or an explicit
pre-run missing reason. Do not remove baselines by target ranking before formal freeze.

### E. Protocol Freeze and Formal GPU Cohort

- Formal freeze remains unreleased. Original `semantic_main_table_freeze_v1` compared
  unmatched phase-gain actors with old task-only and does not establish mechanism gating.
  Its eight-task index also references executed four-task v2 requests, so the entire
  index cannot be called `sealed_unaccessed`. Preserve old records and correct them
  through explicit subsequent revisions.
- Use genuine ICDT and structurally matched ablations to verify directionally consistent
  mechanism benefits across seeds in at least two prespecified critical-contact cells.
  Wins on all twelve tasks are unnecessary. Revise theory claims first if the mechanism
  provides no benefit.
- Choose independent training-instance counts from pilot variance and training cost.
  Eight instances are an existing design reference; a configuration with only a calibration
  seed is not a trained instance. Freeze actual training recipes, selection rules,
  0/30/60/120-second target-interaction accounting, and unaccessed formal requests.
- Use fresh train/development/heldout splits.
- Maintain GPU0-only unless the user updates it. Do not tune midway through formal runs
  or replace failures.

**Exit:** four-metric general table for 12 subtasks × 6 rows, three clean ablations,
three RQ figure sets, hierarchical intervals, and complete budget/latency audits.

### F. Complete Manuscript Evidence

- Link each Introduction contribution to the main table, an ablation, or an event figure.
- Scale conclusions to final evidence and retain failure boundaries.
- Align GIFs, main figures, tables, and appendix provenance.
- Check LaTeX compilation, references, layout, numerical sources, and artifact hashes.

## 13. Planned Paper Figures and Tables

1. **Method overview:** source prior, target intervention, response calibration,
   selective anchoring, target execution kernel, and behavioral events.
2. **Task-complexity figure:** three core subtasks in each of four families, with arrows
   showing added event relations rather than harder parameter values.
3. **General table:** four metrics over six methods × four families/twelve subtasks.
   Main text may use equal-weight family summaries; show all twelve tasks in four
   three-subtask panels without hiding individual failures.
4. **RQ1 figure:** constrained success and heldout response error versus total target
   interaction time.
5. **RQ2 figure:** full/no-calibration/no-response-anchoring/unconditional-response-anchoring
   event success, first force peaks, and recovery timelines in compatible/incompatible
   contact cells.
6. **RQ3 figure:** same-semantics Panda→xArm7/UR5e and Go2+Z1 GIFs/trajectories with p95
   latency and support-failure boundaries.
7. **Failure figure:** first failures in jam overforce, corner contact loss, fixture
   travel/settling, and support infeasibility, alongside successful examples.

## 14. Next Concrete Work

The obstacle family's three-task implementation cycle is now complete: new tasks are
registered, v2 response learning actually ran, paired evidence for both new cells was
re-audited, and the formal release report is `released_for_formal_execution`.
The 100 new formal episodes remain unopened.

Proceed in this order:

1. **Align qualification and evidence.** Add new-protocol qualifications for four existing
   tasks and review compliant contract binding. Explicitly correct phase-gain experiment
   identities and old eight-task freeze credentials, retaining all old evidence.
2. **Execute the released obstacle protocols.** Run two 50-episode protocols in the frozen
   aperture-then-sequential order. Do not replace failures or revise thresholds after
   inspecting outcomes. Reuse completed dual-route formal results from 5 methods,
   8 instances, and 40 requests/instance. After execution, perform only SHA auditing
   and prespecified four-metric aggregation. Two-seed pilots are not final rankings.
3. **Build core contact/recovery evidence.** Complete genuine method controls on existing
   corner/compliant tasks while repairing jam's shared unloading and delayed force
   consequences. Add edge pivot's asymmetric object, pivot, and angular-contact evidence;
   add insert/lock's physical bayonet, retention test, and unlocked negative control.
   The latter introduces new contact topology and requires CPU physics/source validation
   before GPU training.
4. **Complete the support family's third task.** Add replayable external-force pulses
   and a post-disturbance termination chain to existing fixed-foot WBC. Validate both
   maintaining contact and re-establishing it when needed. This family serves morphology
   demonstration and RQ3, without expanding into walking/navigation.
5. **Method qualification, representative-cell pilots, then formal freeze.** Complete
   new-task adaptation and source/nominal qualification for all six methods. Give
   ASID/AdaptSim corresponding task training opportunities; MPC needs relevant angular-
   response, contact, or support rollouts. Answer RQs in representative conditions first,
   then freeze the full twelve-task table and matched-budget curves. Mark unqualified
   implementations pending; they do not establish original-algorithm failure.

Theory still needs event-sufficient responses, finite-phase error composition, and
execution-interface attribution boundaries. Sequential/pivot/lock/disturbance tasks
test those conditions; their addition does not prove the theory. Evaluate selective
anchoring through actual gated/ungated objectives. If independent benefits cannot be
reproduced, narrow the claim instead of manufacturing contrast with aggressive gains.
