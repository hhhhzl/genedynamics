# Cross-embodiment CPU framework

The CPU implementation includes actual 3D execution, source-state projection,
small learned priors/response ensembles, budgeted context fitting, stochastic
transport, global risk reduction, resumable training, and physical evaluation.
This is a research prototype. Small CPU runs qualify functionality, not statistical
method superiority. Current results and retained failures are in the
[validation record](../../../docs/development/cross-embodiment-cpu-obstacle-transfer-validation.md).

## Public components and backend boundaries

| Frontend | JAX CPU | Torch CPU | C++ CPU |
| --- | --- | --- | --- |
| `CrossEmbodiment` | MJX compiled execution, heterogeneous mechanisms, execution microbatch | separate physics adapter required | separate physics adapter required |
| `ContactRunner` | legacy MJX fixed/planar contact; matched keyway/TCP wrench currently NumPy only | separate physics adapter required | separate physics adapter required |
| `LearnedModel`, `FrozenRegressor`, `GaussianSkillPrior` | forward, gradients, training chunks | same portable weights, gradients, CPU update loop | whole-batch inference only |
| `StochasticTransport`, `fit_context` | compiled updates, global CVaR/microbatch | numerical reference, full batch | unsupported, rejected explicitly |
| `CrossEmbodimentSolver` | fused query→prior→transport→MJX outer scan | provider extension point | provider extension point |

NumPy/MuJoCo is the serial float64 diagnostic physics reference. JAX/MJX is the
FP32 device kernel, including when another solver enables x64. GPU selection
fails when unavailable; no silent CPU fallback. Torch/C++ learner portability
does not imply those runtimes implement physics or whole-body GPU execution.
Native ABI v1 has caller-owned contiguous FP32 tensors, `in_out` weights,
version/error checks, no retained pointers, and inference-only capability.
Rust can implement the same ABI later; no Rust implementation is claimed.

`mating_contact_config` qualifies a separate matched-keyway protocol with signed
tool wrench at the TCP, full-tool alignment, and axial support events. New
physics semantics explicitly reject unqualified backends. `PhysicsTwinResponse`,
contact identification, `ContactPlanner`, and `ResearchAudit` use lazy solver-style
NumPy providers. The planner owns candidate/horizon execution and restarts from
encoder q/qdot plus known controller carry, without copying the hidden model.
Stage-conditioned portable affine maps run on JAX, Torch and native C++;
this does not qualify GPU physics. See the
[six-part theory feasibility audit](../../../docs/development/cross-embodiment-theory-feasibility.md)
for commands, evidence and go/no-go conditions.

The matched-keyway study records same-model task-only search tying the transport,
extra noise showing no observed gain, a learned episode-response validation
rejection, and a payload-mismatch rejection. Its fresh 25-reset-cluster cost
audit is conditional on one calibration. The UR5e map holdout concerns this
new map search; it does not retroactively make older shared-response datasets
unseen-body experiments. The latest
[CPU mechanism closure](../../../docs/development/cross-embodiment-validation-closure.md)
adds state-conditioned short/multistep neural response qualification, a separately
scoped finite joint response law, five independent calibrations and optimizations,
full 3D obstacle/guide studies, and coupled Go2+Z1 standing manipulation with the
same frozen insertion prior. The episode-mean cost network remains unsupported;
strong task-only baselines often tie. Local experiments, tests and installations
were stopped at the user's request after disk exhaustion. The revised
body/mechanism matrix is incomplete after a retained matched-xArm failure;
current full JAX/Torch/MJX regression and final acceptance remain pending.
See the [GPU handoff](../../../docs/development/cross-embodiment-gpu-handoff.md)
for saved checkpoints, storage requirements and provider qualification order.
Historical passes are not current qualifications.

Environment names are `obstacle_transfer`, `guided_contact_alignment`,
`keyed_insertion_recovery`, and `mobile_keyed_insertion_recovery`. Panda is the
nominal source; xArm7 and UR5e provide actual 7/6DoF target models. Contact
fixtures use a long common tool and actual solver constraint forces. Positive
command channel 12 means compressive normal force, with world +z normal.
Commands and measured wrench are separate. Terminal status is sticky, fires once,
and padded controls are invalid; source-support failures stay in the denominator.

The planar mobile adapter has x/y/yaw drive joints, ground-contact caster proxies
and constrained height/roll/pitch. Its body intent is known fixture motion, not a
learned navigation policy. The Go2+Z1 adapter has a real floating base, legs,
arm, and feet; a CPU OSQP stance inverse-dynamics controller runs every 4ms.
Foot anchors, body height, QP warm start/residual and clocks are explicit carry.
It supports standing manipulation, not gait generation. Host OSQP is never
presented as a GPU-resident controller. The fused policy currently executes its full world batch; execution-only and training risk microbatching have separate tests.
Humanlike-body integration awaits an
actual selected manipulation/control interface.

## Data permissions and replay

`data.py` and `contact_data.py` separate nominal Panda `source_demonstrations`
from public `response_development` and reserved `response_validation` records.
Episode groups stay together across all body/mechanism variants. Prior training
uses actual successful source modes, not requested teacher labels alone. Failure
trajectories and attempted episode counts remain in artifacts. Normalization uses
training rows only. Public mechanism labels condition offline response training;
hidden target observations and transport inputs do not receive those labels.
UR5e included in response development is a **seen-body engineering** test, not
unseen-body generalization.

`HiddenTargetEnv` charges a single world/reset/rollout before execution. Failed
and interrupted attempts count. A persistent ledger preserves pending costs.
After sealing, final evaluation cannot feed calibration. The current short reset
probes are a weak identification baseline; inspect held-out calibration error.
The exploratory CPU target configurations were used during debugging and are
not an untouched statistical test set.

`RandomRequest` keys resets by sample identity. `random_streams.py` defines
independent named mode/prior/transport streams keyed by seed, identity and window;
providers also accept explicit noise arrays for cross-runtime replay. Source
query accepts task semantics only, constructs its own nominal Panda physics and
fresh controller state, and rejects unsupported pose/twist/contact phases.

Response ensembles predict next complete carry plus risk diagnostics. The
floating extension adds foot anchors, body height, QP warm start and residual;
fixed/planar bodies use declared zero padding for that extension. Autoregressive
rollout advances each member's own carry and retains sticky predicted risk.
Disagreement and risk heads are **uncalibrated surrogates**, not safety certificates.
Clearance in ObstacleTransfer is contact-derived and censored at 5mm. Robot
collision proxies/exclusions are recorded and still require deployment audit.

Artifacts contain `metadata.json` pointing to a content-addressed
`tensors-<sha256>.npz`. Checksums, tensor names/shapes/dtypes and finite values
are validated without pickle. Atomic metadata publication retains the preceding
generation on interruption; old `tensors.npz` bundles remain readable. Inference
weights and `training_resume` checkpoints are distinct. Resume checkpoints include
Adam moments, step, context, multipliers, RNG counter and consumed probe budget.
Training CLI `--resume` checks data/model/random contracts. A resumed contact loop
reuses charged probe records, skips pending/lost attempts without refunds, and
rejects a changed calibration configuration.

## Run CPU qualification

Use the existing image from the repository root:

```sh
docker run --rm -v "$PWD:/workspace" -w /workspace \
  -e PYTHONPATH=/workspace -e JAX_PLATFORMS=cpu -e MUJOCO_GL=osmesa \
  -e XLA_FLAGS=--xla_disable_hlo_passes=constant_folding \
  -e JAX_COMPILATION_CACHE_DIR=/workspace/tmp/cross_embodiment/jax_cache \
  --entrypoint python genedynamics/dev-cpu:local \
  -m scripts.cross_embodiment.qualify_cpu --output tmp/cross_embodiment/qualification_new
```

The runner serializes modules in separate processes and writes logs, JUnit,
versions, source fingerprint and flags. This releases compiled MJX graphs between
modules. Concurrent large-graph CPU compilation caused an OOM during development;
serialize device qualification as well. The XLA flag disables one expensive CPU
compiler optimization, **not JIT**; the library never changes global flags.

For existing learned artifacts add `--device-chunks` for strict transfer guards,
or `--learned-loops` for new charged calibration runs and physical baselines.
Use a new output path for independent target budgets. Inspect each task report:
a valid retained physical failure is distinct from a software exception.

## Data, training and physical-loop commands

Run these within the same CPU environment (`python -m scripts.cross_embodiment...`).
All output paths below are examples; choose fresh paths or explicitly resume.

```sh
python -m scripts.cross_embodiment.collect_obstacle_transfer_data \
  --backend numpy --role source_demonstrations --full-length \
  --output tmp/cross_embodiment/obstacle_source_new
python -m scripts.cross_embodiment.train_cpu_models \
  --source tmp/cross_embodiment/obstacle_transfer_source_numpy \
  --response tmp/cross_embodiment/obstacle_transfer_response_full \
  --excitations tmp/cross_embodiment/response_excitations \
  --updates 5000 --output tmp/cross_embodiment/obstacle_models_new
python -m scripts.cross_embodiment.cpu_learned_closed_loop --help

python -m scripts.cross_embodiment.collect_contact_data \
  --task guided_contact_alignment --role source_demonstrations \
  --output tmp/cross_embodiment/guided_source_new
python -m scripts.cross_embodiment.collect_contact_data \
  --task guided_contact_alignment --role response_development --excite \
  --output tmp/cross_embodiment/guided_response_new
python -m scripts.cross_embodiment.train_contact_models \
  --source tmp/cross_embodiment/guided_source_new \
  --response tmp/cross_embodiment/guided_response_new --updates 5000 \
  --output tmp/cross_embodiment/guided_models_new
python -m scripts.cross_embodiment.cpu_contact_learned_loop \
  --models tmp/cross_embodiment/guided_models_new \
  --response-data tmp/cross_embodiment/guided_response_new \
  --task guided_contact_alignment --budget 10 \
  --output tmp/cross_embodiment/guided_loop_new
```

Use `keyed_insertion_recovery` for the fixed connector task. Collect mobile
responses with `--task mobile_keyed_insertion_recovery --robots xarm7` or
`--robots go2_z1`. `train_contact_response` extends the response model while
copying the frozen Task 3 prior artifact byte for byte:

```sh
python -m scripts.cross_embodiment.train_contact_response \
  --source-prior tmp/cross_embodiment/keyed_models_cpu_final/source_prior \
  --responses tmp/cross_embodiment/keyed_response_v3 \
              tmp/cross_embodiment/mobile_keyed_response_v4 \
              tmp/cross_embodiment/go2_z1_response_cpu \
  --output tmp/cross_embodiment/mobile_models_new --updates 5000
python -m scripts.cross_embodiment.cpu_contact_learned_loop \
  --models tmp/cross_embodiment/mobile_models_new \
  --response-data tmp/cross_embodiment/go2_z1_response_cpu \
  --task mobile_keyed_insertion_recovery --robot go2_z1 --budget 10 \
  --output tmp/cross_embodiment/go2_loop_new
```

`cpu_contact_smoke` is an independent scripted physics diagnostic. Actual
learned-loop evaluation uses direct/uncalibrated/deterministic/full baselines,
the same source prior, initial modes and shared random streams. `evaluate_response`
evaluates reserved public mechanism combinations and never updates a model.
`build_native` builds the C++ library explicitly; set
`CROSS_EMBODIMENT_NATIVE_LIBRARY` before loading `backend="native"`.

GPU, native CUDA/FFI streams, Torch CUDA/DDP and true multi-GPU scaling remain
separate qualification work. CPU tests cannot establish their performance.

## Restricted mechanism revision

The event-conditioned `PhaseGaussianPrior`, `PhysicsTwinResponse` and
phase-conditioned `ReferenceTransport` provide a separate CPU research gate.
The public `research_contact_config` freezes 16mm insertion geometry, 2mm/2deg
acceptance, 0.3s hold, 15s timeout and substep 25N/10N total/lateral force limits.
Historical defaults remain readable. Known body and nominal payload are assumed;
only bounded gain and integer delay are identified. Five charged probes include
contact and independent validation. The twin is a NumPy diagnostic provider;
portable affine/prior inference runs through the actual JAX/Torch/native APIs.
This route does not qualify a neural world model or GPU twin optimization.

Example commands (inside the configured CPU environment):

```bash
python -m scripts.cross_embodiment.cpu_mechanism_validation --research --seeds 40 41 42 --output tmp/revision/source
python -m scripts.cross_embodiment.train_phase_prior --source tmp/revision/source --output tmp/revision/prior
python -m scripts.cross_embodiment.validate_mechanism_chain --prior tmp/revision/prior --output tmp/revision/chain --seeds 8000 8001 8002 8003 8004
python -m scripts.cross_embodiment.complete_chain_baselines --prior tmp/revision/prior --chain tmp/revision/chain
```

Use new output directories for new studies. Calibration resume checks the full
config/mechanism/seed contract and never replays a charged missing attempt for
free. Final task evaluations are separate from the five-probe adaptation budget;
all simulator hypothesis/transport calls and failed task evaluations are reported.
