# Learning and generative inference

GenerativeDynamics connects learned policies and trajectory models to
model-based generative optimization and closed-loop control. Choose the
learning path that matches your task:

| Path | Learn | Use at runtime |
| --- | --- | --- |
| **RL policies** | PPO or SAC in a Brax/MJX environment | A standalone policy controller |
| **MGA policy priors** | PPO on the contact task's training domains | State-conditioned horizon proposals, followed by online generative refinement and acceptance |
| **Learned trajectory diffusion** | DPCC diffusion on offline D3IL demonstrations | Trajectory sampling with DPCC projections or SafeDiffuser CBF corrections |

Model-based diffusion planners can generate trajectories without a learned
checkpoint. Learning and model-based inference are complementary components;
their requirements depend on the selected [planner](../reference/planners.md).

Run the commands below from the repository root in an isolated Python
environment. Check [installation](../getting-started/installation.md) and
[device support](../reference/compatibility.md) before selecting CPU or CUDA.

## Train and reuse an RL policy

The arm training entry supports PPO and SAC. It reads task, training domains,
action transforms and training settings from `metadata.training.rl` in a task
base configuration.

```bash
python -m pip install -e ".[simulation,optimization]"
python scripts/tasks/robot/arm/train_rl_baseline.py \
  --config configs/arm/peg_insert/_base.yaml \
  --algo ppo --out results/learning/peg_insert_ppo.pkl
```

This uses the task configuration's training distribution and budget. Evaluate
the resulting policy before comparing task performance; retraining does not
recreate the frozen paper checkpoint byte for byte. The same entry supports
`--algo sac` for standalone policy baselines.

For a pipeline check, use a separate small run:

```bash
python scripts/tasks/robot/arm/train_rl_baseline.py \
  --config configs/arm/peg_insert/_base.yaml \
  --algo ppo --smoke --cpu-low-memory \
  --out results/learning/peg_insert_smoke.pkl
```

`--smoke` reduces training steps and domains; `--cpu-low-memory` selects a
compact network and PPO batch. This checkpoint checks the training pipeline
and is not the policy used in the MGA recipe below.

Checkpoints retain policy parameters, network configuration, observation
normalization and task/action metadata. Reconstruct policy inference with the
saved configuration:

```python
from genedynamics.learning.train_rl_policy import load_policy, build_policy_act

params, network_config = load_policy("results/learning/peg_insert_ppo.pkl")
act = build_policy_act(params, network_config, deterministic=True)
# action = act(observation_from_the_matching_task, rng_key)
```

Use the same task's observation and action semantics at training and inference.
For surface scanning, use `configs/arm/surface_scan/_base.yaml`; its training
settings include material/geometry domains and domain randomization. An ATACOM
tangent policy is trained separately with `--atacom` and has its own checkpoint.

## MGA: train a policy and reuse it as a prior

MGA reconstructs a **PPO** policy as a reusable prior. It can roll the policy
forward from the current state, map the resulting control horizon to solver
nodes, and sample additional stochastic policy proposals. The online optimizer
then evaluates and refines candidates using dynamics, task objectives and
constraint geometry before the task's acceptance checks select execution.

For a local experiment, save this inherited configuration as
`configs/local/mga_peg.yaml` after creating `configs/local/`:

```yaml
base: ../arm/peg_insert/main/mga.yaml
name: mga_peg_custom_prior
method_params:
  policy_ckpt: results/learning/peg_insert_ppo.pkl
```

```bash
genedynamics-run configs/local/mga_peg.yaml \
  --device cpu --seed 0 --development-root results/development
```

The parent recipe supplies the contact-task contract, `prior_mode: additive`,
proposal settings and incumbent acceptance. Development output keeps a custom
checkpoint separate from formal evaluation. See
[configuration inheritance](configuration.md#inheritance) for composing
variants. Task performance requires a trained and evaluated prior. Keep the
pipeline-check checkpoint separate from this experiment.

The shared API is `build_policy_prior(params, config, env=..., Hsample=...,
Hnode=..., ctrl_dt=...)` in `genedynamics.learning.train_rl_policy`. A matching
PPO checkpoint is required for this sequence-prior adapter. SAC currently
supports standalone policy inference rather than MGA horizon priors.

## Learned trajectory diffusion

DPCC uses a Torch diffusion model trained on offline demonstrations. In
addition to the Python extras, this source-tree workflow needs the vendored
`third_party/diffuser` package, D3IL environment assets and the avoiding dataset.

```bash
git submodule update --init --recursive
python -m pip install -e ".[simulation,d3il,torch]"
export PYTHONPATH="$PWD/third_party:${PYTHONPATH:-}"
export DPCC_AVOIDING_DATA_DIR="/absolute/path/to/avoiding/data"

python -m genedynamics.solvers.single.dpcc.tools.train \
  --config-file configs/d3il_avoiding/dpcc_diffusion_train.py \
  --dataset avoiding-d3il --seeds 5 --device cuda
```

Set the dataset directory to the extracted D3IL trajectory files. The 4D
loader consumes each file's `robot.des_c_pos` and `robot.c_pos`; it derives
actions from consecutive desired positions. Keep this directory available
during evaluation because the loader reconstructs the dataset normalizer.
This training command targets a CUDA Torch installation on a compatible host.

The supplied configuration writes training configurations, checkpoints and
losses under `logs/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion/5/`.
Evaluate the matching seed and best checkpoint with:

```bash
genedynamics-run configs/d3il_avoiding/dpcc.yaml --seed 0
```

Here `method_params.seed: 5` selects the diffusion model, while `--seed 0`
selects the experiment seed. The recipe also declares the model path, horizon
and normalization setup. DPCC applies model-based projections during sampling;
`configs/d3il_avoiding/safediffuser.yaml` loads a compatible diffusion checkpoint
and applies CBF/QP corrections during denoising. See the
[D3IL integration](../integrations/d3il.md) and
[planner reference](../reference/planners.md) for the environment and variants.

## Calibrate task-specific reliability

Surface scanning also uses a learned reliability model to estimate execution
risk from observable state/action features. Fit and calibrate it on disjoint
completed run outputs, then report performance on a third held-out split:

```bash
python scripts/tasks/robot/arm/train_mga_reliability.py \
  --training-results results/learning/scan-fit \
  --calibration-results results/learning/scan-calibration \
  --test-results results/learning/scan-test \
  --quantile 0.95 --output results/learning/reliability_q95.json
```

Replace those three example directories with collected unified-runner outputs
containing `results.json` and their trajectory files. This command fits the
predictor and calibrates upper risk bounds; it does not collect trajectories.
New fits remain development artifacts until independently validated. The
surface-scan recipe declares a frozen reliability checkpoint and abstains to
the task's model-based certificate outside learned support.

The main peg-insertion recipe instead sets `learned_reliability: false`; it
still uses a learned PPO prior and model-based acceptance. Its reliability
collection/validation configurations are separate development paths. A learned
policy prior and a learned reliability gate have different roles and can be
enabled independently by the task recipe.

## Optional diffusion-prior extension

`genedynamics.learning.priors.diffusion.LearnedDiffusionPrior` provides a JAX DDPM
score-MLP with `train`, `score` and sequence `warm_start` methods; `EliteBuffer`
stores reusable trajectory samples. These are extension components. Published
MGA recipes use PPO priors, and the current MGA transport passes
`score_g=None`: it does not automatically train an elite-buffer diffusion model
or fuse that model's score into the online solver.
