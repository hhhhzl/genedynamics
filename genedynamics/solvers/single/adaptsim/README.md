# AdaptSim solver

`AdaptSimSolver` implements the task-driven simulator-parameter distribution
update and native PyTorch branching-Q learner from the [official AdaptSim
release](https://github.com/irom-princeton/AdaptSim) at commit
`9cf80fb4268749fddf21c39eb861498bc7b052c3` (MIT license in `LICENSE`).
The parameter action map follows upstream `parameter_base.py`; Q uses the
upstream value/advantage decomposition and one-step Huber or temporal-difference
loss. The task must still supply simulation rollouts, policy training under each
sampled parameter distribution, target reward/features, and interaction counts.
The method-local `learning` loop can call these task operations and accounts
for every simulated training step and target rollout. Its online Q update is
restricted to simulated meta-targets and is a local adaptation of the upstream
replay schedule; deployment target rewards cannot update the inference network.
`AdaptSimMetaTraining` now trains the Q updater on separately sampled simulated
targets, with a replay buffer and explicit policy-training and rollout counts.
It reproduces the source's inner simulated-target train/evaluate/update
sequence. The outer loop now evaluates all particles on a fixed simulated
meta-target, conditionally replaces the worst with a perturbed copy of the
best, and retains the inference replay buffer across iterations. Gaussian
particle perturbations are numerically checked against the pinned author
code. Environment-specific reward scaling and the author's task-policy
optimizer remain task adaptations.
The solver is an adapted method core. The T1 xArm7 binding has a complete
multi-particle development run and seed-disjoint source/nominal diagnosis.
T2 has a three-gain public guide-feedback trainer. T3 has a seven-parameter
public-feedback trainer and a completed development chain. T4 uses the same
seven-parameter policy family with public body intent and a four-dimensional
mechanism distribution; its population training is a separate task-local run.
These single-instance development chains are not yet a scored four-task
baseline.

The author's pushing example uses a one-step 2D task policy. The current
T1--T4 protocol uses 13D actions and long episodes. A T1 development binding
now trains a six-parameter path/velocity task policy by bounded CEM under each
sampled simulator distribution, executes both public modes, and reports all
simulation and target interactions. This is an explicit task-policy optimizer
adaptation, not the author's one-step NAF. Its T3 target cells also include
discrete control delay, which the upstream parameter update explicitly does
not implement. A four-task port must report these task adaptations and cannot
silently add a delay branch or substitute a task reference for a trained
AdaptSim policy.

The deployment loop can now train each initial simulator particle and compare
their policies on the same target reset before adaptation. Selection uses only
the observed target reward; all particle training and target rollouts are
charged separately, and the chosen policy warm-starts subsequent task-policy
training. The T1 binding has exercised two outer meta iterations on GPU1, but
both particle rewards tied in both iterations and no replacement was triggered;
the replacement branch is tested synthetically. That development run preceded
the persistent-replay correction and remains an integration check rather than
nominal qualification. A full meta-training budget is still outstanding.

The task-policy search now optionally evaluates symmetric, task-declared
coordinate probes alongside random CEM candidates. This is a long-horizon
task optimizer adaptation, not an author-native component. In a separate T1
development run, the probes produced distinct trained policies and separated
two particles on the same deployment reset. That observation validates the
task signal for particle selection; it does not establish convergence of the
inference Q network or four-task baseline performance.

The T1 ledger replay retains outcomes for all 588 candidate episodes, including
82 failures. Its final policy and Q weights match the earlier run. A separate
48-episode, 24-pair source/nominal/target diagnosis records raw physical
evidence: both the adapted policy and public anchor succeed on every request,
while the adapted policy is slower on target mode 0. These are development
results from one trained instance, not evidence of four-task superiority.

The first T3 task-policy trainer evaluates the same bounded public feedback
actor under sampled gain, payload and delay mechanisms. Its 72-episode
development run and a separate 96-episode held-out diagnosis retain every
failure and paired physical reset. AdaptSim succeeds on 11/16 source, 8/16
nominal and 4/16 high-delay requests; the matched public anchor succeeds on
12/16, 7/16 and 4/16. The run does not qualify this single trained policy as
a strong T3 baseline. Both actors frequently exceed the early-contact force
limit, so the common contact condition needs a separate solution before
larger method comparisons.

No AdaptSim or ASID results should enter the independent four-task table before
full method qualification on the frozen public protocol.
