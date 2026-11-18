# Four-Environment Strong-Baseline Comparisons and Paper Contribution Plan

Updated: 2026-10-16. The machine-readable contract for the cross-task main table is frozen at
`configs/cross_embodiment/four_task_benchmark/protocol/main_table_v1.json`; where this document's earlier
task-specific candidate descriptions conflict with that contract, the machine-readable contract takes precedence.
The user clarified that “dense” means **mature, complete, strong baselines**, and added the requirement to prioritize
leading-conference papers from the past two years, preferably including work by Karen Liu and Guanya Shi.
First establish the actual performance of competing methods across the four environments, then determine the incremental benefit our method and theory should explain.
The implementation sequence from P3 onward has been renumbered P3–P7; see the
[phase execution plan](cross-embodiment-p3-onward-plan.md). This document continues to define method candidates and comparison protocols;
phase numbering and completion criteria follow the new plan. Training had not started at the initial draft; native/adapted
training pipelines have now been implemented and executed. See the [P3 execution record](cross-embodiment-p3-execution.md) for the latest facts. Qualification training does not mean a strong method has been trained sufficiently.

## 1. Revised Research Sequence

The next phase does not center on expanding finite mapping enumeration, upgrading the flow, or adding source-anchoring weights.
The revised sequence is: **freeze comparable tasks → integrate strong baselines → establish development-set performance and cost curves
→ identify contributions to test → revise the method and theory → freeze independent main experiments**.

Existing P0–P2 artifacts serve as engineering references and mechanism diagnostics. The full/task-only comparison over finite affine maps
remains an internal ablation; it cannot represent the mature capabilities of continuous policies, complete identification systems,
or online model-based planning. Final target evaluation continues to use the independent CPU reference; GPU search qualification does not automatically extend
to new policies, tasks, contact topologies, or WBC.

See the [baseline survey](cross-embodiment-baseline-survey.md) for literature and native-implementation verification.
See the [four-task protocol](cross-embodiment-four-task-protocol.md) for interfaces, current status, and code conflicts.
In both documents, “implemented” refers only to explicitly listed code and qualifications, not to a method having run all four tasks.

## 2. Prioritize Recent Main-Conference Papers and Verify Both Authorship and Reproduction Scope

Main candidates primarily come from **2025/2026 main-conference papers at CoRL, RSS, ICRA, ICLR, NeurIPS, and similar venues**,
prioritizing complete methods directly relevant to the tasks and coauthored by C. Karen Liu or Guanya Shi.
For accepted 2026 papers whose proceedings have not yet appeared, record the acceptance evidence; do not pass off an arXiv year,
a workshop paper, or a journal paper as main-conference status. As of 2026-10-14, conferences in late 2024 remain within the rolling
two-year window but are boundary candidates only; PMLR publication in 2025 does not turn CoRL 2024 into CoRL 2025.

| Main candidate | Formal status and requested author | Competing explanation and task role | Integration priority/boundaries |
| --- | --- | --- | --- |
| **FADA** | Accepted at CoRL 2026; Guanya Shi; verified through author announcements and official repository | Freeze the planner and adapt IDM using only target actions/observations; first rule out ordinary execution adaptation as a sufficient explanation | Initial-round B-log. Native setting is same-body humanoid dynamics adaptation; T1–T3 require retraining task oracle/Planner-IDM, and T4 needs interface qualification. This is not a ready-to-use cross-arm checkpoint |
| **Human2Sim2Robot** | CoRL 2025, formal PMLR; Karen Liu | Are object-centric demonstration rewards and target-model RL enough for cross-embodiment task completion? Prioritize T1/T2 | Initial method route. Separate native human RGB-D video supervision from adaptation using shared source object trajectories; retain digital-twin construction and exploration initialization |
| **ACP** | ICRA 2025; Shuran Song et al., neither requested author | Complete contact policy with reference poses, virtual targets, and variable stiffness; T2/T3 | Initial contact control, requiring force demonstrations and classified as C-supervised; provides a direct compliance competitor missing from a pool restricted to the two requested authors |
| **Play2Perfect** | Accepted at CoRL 2026; Karen Liu; verified through author/Stanford IPRL announcements | Multi-object play pretraining and precise-assembly RL fine-tuning; T3 first, T2 second | Audit native/adaptation eligibility before deciding main-table admission. Native dexterous in-hand control is central; a fixed-grasp interface cannot remove it and still claim the complete original method |
| **ASAP** | RSS 2025, formal proceedings; Guanya Shi | Learn a delta-action model from B-log target trajectories and fine-tune the source policy in the augmented simulator; rule out conventional simulator alignment as sufficient | Key adaptation control in the second batch. T4 is close to the original; fixed-arm settings are explicit adaptations. Declare complete target records, state alignment/additional measurements, delta model, and policy fine-tuning |

Primary factual sources: [FADA acceptance announcement](https://angchenxie.github.io/),
[FADA official code](https://github.com/LeCAR-Lab/FADA),
[Human2Sim2Robot formal paper](https://proceedings.mlr.press/v305/lum25a.html),
[ACP](https://adaptive-compliance.github.io/),
[Play2Perfect acceptance announcement](https://iprl.stanford.edu/),
[Play2Perfect official code](https://github.com/kushal2000/play2perfect),
[ASAP formal paper](https://roboticsproceedings.org/rss21/p066.html).

The initial round focuses on **FADA, the Human2Sim2Robot task-adaptation route, and ACP**. These compare execution
adaptation, target-policy learning driven by source semantics, and compliant contact control, respectively. Their access includes B-log, source-trajectory adaptations/
native human videos, and C-supervised, so they cannot be combined
into one equal-budget ranking. First determine whether Play2Perfect can retain its core within the current fixed-grasp tasks;
add ASAP when simulator alignment is confirmed as a major competing explanation. The formal main paper should converge on
3–5 faithfully reproducible methods; a leading-conference/author label does not itself confer main-table eligibility.

### 2.1 Recent Complete Methods Added by Task

- **SPI-Active (CoRL 2025, Guanya Shi)**: prioritize Go2 active identification in T4.
  The official release includes SPI, active exploration, and downstream training, with gaps in data connections/hardware;
  do not claim arbitrary arm/contact parameters work out of the box. [Formal paper](https://proceedings.mlr.press/v305/sobanbabu25a.html),
  [official code](https://github.com/LeCAR-Lab/SPI-Active).
- **DIAL-MPC (ICRA 2025, Guanya Shi)**: recent strong planning candidate. When optimizing task references with the same calibrated model,
  label it planner-adapted; native whole-body torque-level control receives a separate complete-system comparison.
  Replacing its core with the shared old WBC cannot be called native. [Official code](https://github.com/LeCAR-Lab/dial-mpc).
- **XMoP (ICRA 2025)**: direct cross-embodiment obstacle-avoidance planning reference for T1; official code/models/data
  are public. Report large-scale procedurally generated planning data separately; carried-object collisions and dynamic mechanisms still need adaptation.
  [Author project](https://prabinrath.github.io/xmop/), [code](https://github.com/prabinrath/xmop).

Karen Liu's **SimToolReal (RSS 2026)** and **Cloak (CoRL 2026)** remain candidates:
the former retains a dexterous-hand degrees-of-freedom boundary, while the latter needs a complete visual/VLA and external-pretraining protocol;
its core visual masking cannot be removed to place it in the state/force main table. SPARR (ICRA 2026) and LAD (ICRA 2026)
are highly relevant, but a complete public execution chain was not verified in this review; they await release/faithful-reproduction qualification.

### 2.2 Retain Basic References and Ablations Outside the Recent Main-Conference Ranking

DR/PPO, classic Residual RL, MuJoCo MPC, ASID (ICLR 2024), and Mirage (RSS 2024)
remain engineering sanity checks and references for ordinary calibration, control, or historical methods, rather than occupying
main-baseline slots for recent papers. **FORGE is RA-L 2025; recency does not turn it into a leading-conference main-track paper**,
although it remains a strong contact reference. Retain Direct, classic hybrid force/position control, and force-limited search.

Capacity-matched task-only, deterministic transport, removal of response/mode/risk,
reweighting-only, and arm-only/coupled calibration are internal ablations. Report these separately from mature external
methods, and do not restrict all competitors to the existing 24 affine maps.

## 3. Comparing the Four Environments

| Environment | Recent-paper candidates and comparison roles | Main performance and incremental question |
| --- | --- | --- |
| **T1 ObstacleTransfer** | Human2Sim2Robot source-trajectory adaptation, FADA/ASAP execution adaptation; XMoP and DIAL-MPC geometry/planning comparisons | Full-arm/carried-object collision-free completion, SE(3) error, clearance, time. Is there added value beyond ordinary source-semantic RL, geometric planning, and dynamics alignment? |
| **T2 GuidedContactAlignment** | ACP demonstration-supervised track; Human2Sim2Robot object-trajectory track; FADA/ASAP execution adaptation; Play2Perfect after qualification | Force-limited positioning, contact maintenance, slip/stall, force peaks. Does response calibration improve on sufficiently trained contact feedback and compliance learning? |
| **T3 KeyedInsertionRecovery** | Play2Perfect qualification priority, ACP, FADA/ASAP; SPARR joins B-online after code/reproduction qualification | Full-denominator insertion, jam recovery, retries, excessive force, rejection, and timeout. Have conventional RL after pretraining or compliant policies already solved recovery? |
| **T4 MobileKeyedInsertionRecovery** | FADA/ASAP body adaptation; SPI-Active Go2 active identification; separate DIAL-MPC native-system and shared-WBC adaptations | Joint success of insertion and support, slip, WBC/body failures, end-to-end latency. Is coupled calibration useful under equal control capabilities? |

Methods ultimately admitted to the main table must cover all four tasks; an unexecuted method cannot be recorded as a failure, zero, or N/A.
v1 selects FADA-task-adapted, ACP-task-adapted, Human2Sim2Robot-source-trajectory,
and DIAL-MPC-task-adapted, retaining ordinary response MPC, public geometry, and public
source prior as three strong public controls. Each row explicitly reports data access. Cross-permission comparisons describe only the resource–performance
frontier; algorithmic superiority requires equal permissions. Native complete-system evidence has a separate identity;
an adaptation retaining only a small module cannot stand in for the original paper. Play2Perfect, XMoP, SPI-Active, and ASAP are not yet called baselines
in this version; they may enter targeted paper analysis only after four-task adaptations that preserve their core mechanisms are complete.

Establish comparison entry points for T1/T3 first, while qualifying T2/T4 environments and interfaces in parallel. Resolve current draft/code conflicts in
time limits, force limits, and recovery labels on the development set before freezing the formal protocol. A planar-caster
proxy cannot support conclusions about real wheel–ground stability; the current standing Go2 + Z1 WBC does not imply navigation/gait capability.

## 4. Separate Tables by Data Access

| Protocol | Permitted target information | Training procedure | Supported comparison |
| --- | --- | --- | --- |
| **A-fixed: identical short-probe buffer** | The same charged probe buffer and known geometry/interfaces, without full-task demonstrations | Method-specific model/context fitting; policy updates only in calibrated models or public development domains | Isolate incremental benefits from identification, policy learning, and response objectives |
| **A-active: complete active system** | Each method chooses probes; total count, duration, and failures charged consistently | Retain the full active exploration → identification → in-model training chain | Compare native active-identification systems without pretending they share an identical buffer |
| **B-log: complete self-collected target records** | Charged complete action/observation trajectories; no target rewards, successful expert demonstrations, or corrections | Native FADA adaptation, ASAP data alignment, etc.; declare extra state measurements individually | Compare execution adaptation with equal trajectory/observation access, separately from A short probes |
| **B-online: target-task adaptation** | Explicitly charged complete task rollouts and rewards, without extra demonstrations | Native target-online residual RL, fine-tuning, etc. | Performance curves against target-task transitions/minutes |
| **C-supervised: extra-supervision reference** | Successful target demonstrations, human corrections, aligned actions, or external pretraining, itemized | Native supervised training; target online interaction has its own charged column | Reference for achievable performance with more resources, not ranked alongside A |
| **D-oracle: sealed ground-truth diagnosis** | True mechanism parameters, ground-truth-model planning, or high-budget feasibility search | Post-freeze diagnosis only; results do not feed back into training | Explain mechanisms/support, not an equal-budget main ranking or mathematical upper bound |

Task goals, public training rewards, and known body models are not additional secret information.
The main protocol hides the actual execution mechanism and complete target-task training data.
B-log's lack of rewards does not make it A: trajectory coverage and access to state information also differ.
Control methods in the same group receive identical available records/measurements. FADA/ASAP versions using only short probes are labeled
probe-adapted and do not represent their native complete-data protocols.
In-model rewards come from calibrated models; hidden target ground truth cannot be queried as a free GPU training environment.
A privileged critic during training may use states from its public/calibrated training model; the deployed actor reads only
public observations, and hidden target parameters are forbidden in training any main method.

“Requires no target demonstrations” does not mean “uses little target data”: large-scale random interaction, for example, must still be charged separately.
Policies trained on a known nominal target body and frozen unseen-body generalization are also reported separately.

## 5. Fairness and Reproducible Implementation

### 5.1 Equal Information and Control Capability, Without Reducing Strong Methods to Candidate Selectors

- Share measurable task state, wrench, equal-length histories, known geometry, control period, command bounds,
  and low-level decoder/WBC. If Full uses hand-designed phase/event features, baselines receive the same features,
  or they are removed for everyone. Warm starts/hidden delay queues in physics checkpoints are not actor observations.
- Mature methods may retain suitable network architectures, continuous actions or chunks, and full online planning.
  Disclose parameter counts and model-call volume; only internal ablations require matched capacity.
- For methods natively outputting restricted poses, retain a native action-subspace version; label extensions to 13D
  adapted and examine the effect of adding stiffness/force degrees of freedom.
- Within one state benchmark, use the same reference physics and success evaluator. First check small-scale reproduction of the native algorithm,
  then adapt project interfaces. Do not mix native IsaacLab scores with MuJoCo
  adaptation scores. Porting FORGE does not require first moving the whole project to IsaacLab.
- Save force spikes, failures, and rejections at every physics step. Fix MPC timeout/fallback rules in advance;
  do not count only successful trials where planning finished.

Native FORGE already has an [official IsaacLab implementation](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/forge),
but its force/action/body interfaces differ from this task. The official ASID release mainly demonstrates FI exploration objectives;
a complete four-task pipeline still requires adaptation. These are integration tasks, not completed reproductions.

### 5.2 Record Training and Target Costs Separately

Target access is primarily measured in actual seconds, transitions, and resets; short-probe counts are supplemental.
The original `B ∈ {0, 5, 10, 30}` may remain a short-probe subprotocol. For methods such as FADA with approximately two minutes of native
action/observation records, include data points of corresponding duration; an extremely short, unsuitable buffer cannot represent
complete performance. Freeze per-attempt duration and fit/validation allocation on the development set; interface adjustments, validation failures,
and attempts are all charged. Methods requiring complete task records have separately declared record access, even without rewards;
they cannot silently be counted as short probes. Successful target demonstrations belong separately to C-supervised.

Separately report public source/multi-mechanism data volume, simulation transitions, model rollouts, update steps,
tuning trials, wall time, and GPU time. Plot both equal-simulation-data curves and performance before/after training saturation;
a single small update budget must not leave a mature method perpetually undertrained.
Freeze development-tuning and formal-training budgets in advance; interpret differences only after examining learning curves and failure causes.

For deployment, report end-to-end p50/p95 latency and deadline misses covering all preprocessing, source queries, policy/transport,
model planning, low-level control/WBC, and synchronization. Separate first compilation, steady-state operation, and resets.
GPU-kernel timing or network-forward timing does not replace the complete cost of each robot control decision.

### 5.3 Cross Mechanisms with Bodies and Use Independent Hierarchical Statistics

First cross body and execution response on Panda/xArm7; report contact-friction/compliance changes separately, then test
preregistered combinations. Evaluate UR5e separately for frozen/context-only transfer and in-model few-shot adaptation.
In T4, fixing/releasing the same arm's base only isolates support effects; locked-base cases do not enter the mobile main success rate.

Public development mechanisms support selection of rewards, randomization ranges, model classes, and hyperparameters. Seal formal mechanism combinations,
target initial states, and disturbance lists before method freezing; final target trajectories never feed back into selection or further tuning.
Old CPU artifacts are not reused, and targets already accessed in P0–P2 remain development evidence.

Independent training/calibration instances are the statistical replication level; paired resets within an instance reduce comparison variance.
Plan sample size from development variance and the minimum meaningful effect. The original “5 repeats × 50 episodes/cell”
is only a starting point for resource estimates, not a complete scale promised before runtime measurement.
Report intervals per task and body; normalized four-task aggregation is supplementary and cannot hide a failed task.

## 6. Performance Evidence Needed for Contributions

| Candidate contribution | Main figure/table | Evidence required | Insufficient observations |
| --- | --- | --- | --- |
| **Transfer efficiency with little target data** | Constrained success versus target-probe seconds/count; data needed to reach a fixed success level | Lower target-data needs or greater reliability at equal data against recent strong methods with equal access, across independent calibrations | Only better than uncalibrated Direct; only fewer in-model training samples |
| **Contact/recovery robustness** | Overall success, recovery, excessive force, and timeout under force/body constraints; mechanism OOD | Improvements unexplained by ACP/Play2Perfect, ordinary action alignment, or task residuals | Lower mean force with more jams; only reporting successful-recovery subsets |
| **Task-effective behavior selection** | Pareto relation between task cost and requested-mode completion/feasible-mode coverage/response quality | Improve behavior with clear task meaning within preregistered task tolerances, with competitors receiving the same mode requests | Arbitrarily preserving source proportions; reduced source distance with more rejection or failure |
| **Amortized online execution** | Success versus end-to-end latency, model calls, and deadline misses | Lower online cost at task performance close to calibrated MPC, also compared with strong policy baselines | Faster transport forward pass alone; ignoring WBC or training costs |
| **Body-coupled transfer** | Separate fixed/mobile same-arm, arm-only/coupled, and UR5e frozen/adapted tables | Coupled calibration improves joint task success with the same low-level controller and budget | A stronger WBC was substituted; success only on a planar proxy or static support |

**Mode fairness:** Tasks without mode requests permit any valid successful strategy; task-only cannot be marked failed
for selecting a more reliable mode. When explicit detour/contact/recovery requests exist, every method receives the same request;
then compare requested completion, feasible coverage, and autonomous target completion. Response discrepancy is explanatory,
not a substitute for task performance, and algorithmic rejection is not automatically beneficial.

Preregister specific success thresholds, tolerances, and minimum meaningful improvements after determining them from task needs and development variance.
Do not prewrite expected superiority or search the final data for winning metrics.

## 7. How Baseline Results Guide the Method and Theory

| Development-set observation | Research judgment and next action |
| --- | --- |
| FADA/action grounding already explains the benefit | Check whether cross-contact-phase effects, nonlinear delays, or behavior selection are actually needed; ordinary action alignment cannot be repackaged as novel transport |
| Recent contact policies or DR-RPPO already achieve equal task performance under their data permissions | Test data efficiency, unseen mechanisms, and task-meaningful selection; if none adds value, narrow the method contribution |
| A complete SPI-Active/ASID chain is equally effective | Attribute calibration/active exploration to existing methods; demonstrate the independent value of subsequent behavior transfer |
| Calibrated MPC has high success but high online cost | Test amortized policies under equal latency constraints; also compare with sufficiently trained dense policies |
| Full rejects or fails more often than task-only | Separate model bias, support restrictions, selection rules, and response-objective bias first; do not improve apparent performance by reporting only deployed subsets |
| Stochastic/flow variants do not improve effective performance | Use a simpler mapping; do not claim extra randomness or a generative architecture as necessary contributions |

Retain conditional analysis of calibration error, expressive support, sampling error, and stopping cost.
Once contribution directions are clear, align the main theoretical results with the relevant comparisons: for example, task–response tradeoffs,
feasibility under model error, or the cost relationship between online planning and amortized policies. Theory does not assume Full always wins,
and a small successful experiment cannot establish uniform error bounds or general safety guarantees.

## 8. P3–P7 Execution Sequence

See the [new phase plan](cross-embodiment-p3-onward-plan.md) for detailed tasks, reproduction levels,
completion criteria, the method-improvement loop, and resource estimates; it replaces the old P3.0–P3.3 schedule.

| Phase | Implementation | Exit condition |
| --- | --- | --- |
| P3 | Native reproduction per method, isolated runtimes, interface/data contracts, task-specific adaptation qualification | Qualified recent baselines, complete-chain evidence, and measured resources; smoke tests do not count as reproduction |
| P4 | Four-task development comparisons, T1/T3 first, then T2/T4 after qualification | Strong competitive performance levels, data/compute curves, and failure attribution |
| P5 | Diagnosis-driven improvements to continuous feedback, models/calibration, feasibility/stopping, and coupling | Equal-budget task/efficiency gains and ablations that distinguish contributions |
| P6 | Freeze methods and protocols; new mechanisms/initial states and independent training/calibration repeats | Formal tables/figures, intervals, resources, and full-denominator audit |
| P7 | Complete the paper using the final method and evidence | Traceable claims and evidence; development logs do not flow back into the draft |

Interleave P4/P5; strengthening our method need not wait until every external method finishes. Before formal P6,
complete comparison qualification while retaining all four tasks. Select main candidates by fit/completeness, never by final wins and losses.

**Equal-access main-comparison supplement:** B-log is not merely an extra-resource table for FADA/ASAP. Ours and those methods
receive the same target duration, measurable fields, resets, and reward-free recording access, while choosing their own collection procedures,
to compare complete adaptation pipelines. Also run an exactly shared-record control to isolate collection from adaptation gains.
Retain A-fixed for short-probe mechanism comparisons. Give ours the same demonstrations for equal-access ACP comparisons and label it ours+demo;
otherwise compare only system Pareto frontiers under different resources, without using a win under narrow A permissions to claim overall superiority.

Install new algorithms/frameworks through independently removable dependency profiles, keeping the existing P2 runtime frozen.
Currently, the environment manager's `--name` changes only the directory and does not replace a new baseline installer.
New continuous policies and physics topologies each need qualification. Configuration/results continue to mirror domain/task/category/method/
level/seed; development prefixes follow existing conventions, without introducing `results/<project>/<run_id>/`.

## 9. Separate the Paper from Development Records

`latex/cross_ebdm` retains paper text, methods, general theory, experimental protocols, and finally selected evidence.
This round removed 13 experiment-report attachments and old PDFs; development scores, P1/P2 explanations, engineering
acceptance checks, and execution roadmaps were also cleared from the main manuscript. Original-text snapshots, moved files, and SHA manifests are in the
[development report archive](cross-ebdm-experiment-reports/README.md).

Historical runs and original reports in `results` are not rewritten. Experiment-report exports remain in experiment `_reports` by default;
even an explicit `--report-export-dir` rejects the entire `latex/` tree, and the old `--paper-dir` is only a compatibility argument name.
When introducing formal paper results into the draft, select final figures/tables that support the claims and write scientific conclusions;
do not automatically concatenate development reports.
