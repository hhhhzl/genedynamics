# ACP source integration

Pinned reference: [author repository](https://github.com/yifan-hou/adaptive_compliance_policy),
commit `74d68eca3bfb5cb59da131c6351d9e1c9b129ff6`, MIT. `LICENSE` retains the
author's license; `SOURCE.json` records original file hashes and local adaptations.

This package ports the actual `DiffusionUnetTimmMod1Policy` sparse diffusion
model, including its conditional U-Net, learned normalizers, stochastic training
loss, and native checkpoint parameter names. The trained action remains 19 values
per arm: reference pose9, virtual-target pose9, and scalar stiffness. It also ports
`VirtualTargetEstimator` for demonstration labels and the native deployment's
anisotropic stiffness reconstruction. The public `ACPSolver` is a deployment
interface for this concrete model, rather than a generic action callback.

Instantiate `ACPDiffusionPolicy` with the original observation encoder and
Diffusers DDIM scheduler configured by the author's task/workspace configuration.
Instantiate `ACPSolver(policy, shape_meta, device="cuda:0", ...)` after loading the
matching trained weights. `solver.act(public_observation)` receives:

- `sparse`: unnormalized native sensor histories, shaped `(B, T_key, ...)`,
  already downsampled and expressed relative to the latest tool pose as in the
  author's `sparse_obs_to_obs_sample`. RGB is float `[0,1]` in `(B,T,C,H,W)`;
  wrench is `(Fx,Fy,Fz,Tx,Ty,Tz)` in N/Nm in the native observation frame.
- `current_tool_pose_world`: world-from-tool matrices `(B,R,4,4)` in meters,
  with `R=1` or `2`.

The result contains `(B,H,R,4,4)` reference and virtual-target world poses,
`(B,H,R,6,6)` stiffness matrices in the observation-time tool frame, scalar
stiffness, waypoint times in seconds, and the execution horizon. Translation
stiffness is N/m; rotational stiffness is Nm/rad. The task adapter must execute
the virtual target with this compliance controller, and record the controls
actually applied to the plant. Default sample timing follows the author's runner
(2 ms raw sample, 50-step spacing, slowdown 1.5); each task must explicitly set
its timing before comparisons. A scalar prediction is not silently clipped.

`policy.compute_loss(batch, args)` preserves the actual native training objective.
Training and dataset construction run separately from solver inference. Normalizer
statistics, action ordering and native encoder topology must match the checkpoint.
`load_policy_state_dict` accepts the author's `state_dicts.model` or
`state_dicts.ema_model` tensor payload and loads strictly. `load_checkpoint`
loads tensor-only exports using `weights_only=True`; the author's full
dill/Hydra workspace file should be loaded and exported in its native environment.
EMA selection never silently falls back to another model.

The parity tests execute the pinned upstream estimator, action postprocessor,
stiffness loop and production diffusion class. Policy tests use the same small
encoder fixture on both sides to isolate U-Net, scheduler, loss, gradients,
AdamW update, sampling, native state dictionary and structured solver output.
CPU and CUDA run the same checks via `GENEDYNAMICS_BASELINE_TEST_DEVICE`.

The native route now includes the actual CLIP/force-ViT encoder and the alternative
causal-force encoder, with the pinned `multimodal_representation` code and its own
MIT license. `load_native_policy` builds the exact native topology from an author's
workspace checkpoint and loads its trained vision, force, diffusion and normalizer
weights strictly. A redundant pretrained network download is skipped only when
the complete author checkpoint immediately replaces those weights. Native
qualification uses the real published FFT checkpoint and a real released episode,
not a fixture encoder. It compares original/ported prediction, training loss and
encoder gradients, regenerates all episode labels, and verifies exact continuation
of model, optimizer, accumulated gradients, EMA warmup, LR schedule, RNGs and data
cursor across save/load.

The source checkpoint configs retain two inactive legacy dense flags that the
released code no longer accepts. Only false flags are removed and recorded;
active legacy behavior is rejected. The original optimizer updates diffusion
and RGB backbones while omitting force/fusion parameters. `upstream` mode preserves
that behavior and records omitted parameters; `all_trainable` is an explicitly
adapted variant. Complete continuation checkpoints use tensor-only serialization,
including normalizers and sampler/RNG state; the original workspace did not save
all those continuation fields. Format v2 also saves gradients accumulated on the
89 trainable parameters omitted by the original optimizer. An upstream-mode v1
checkpoint cannot claim complete continuation and is rejected; adapted
`all_trainable` v1 checkpoints remain loadable because their gradients are cleared
by the optimizer after every update.

The independent removable profile is managed by:

```bash
python -m scripts.baselines.acp_native_environment create --gpu 1
python -m scripts.baselines.acp_native_assets fetch --asset checkpoints --output .runtime/acp_native/assets --member 'flip_up_spec/*'
python -m scripts.baselines.acp_native_assets fetch --asset flip_up --output .runtime/acp_native/assets --member 'flip_up_new_v5/.zgroup' --member 'flip_up_new_v5/data/.zgroup' --member 'flip_up_new_v5/meta/*' --member 'flip_up_new_v5/data/episode_1727235551/*'
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cpu .runtime/acp_native/venv/bin/python -m scripts.baselines.acp_native qualify
python -m scripts.baselines.acp_native_environment clean
```

Check out the encoder submodule at `e4bd73e3e6ff0d4334d12aabbe0e9575669b63e7`
in `baselines/ACP_multimodal_representation` before qualification. The native
configuration and result destination mirror
`configs/arm/flip_up/qualification/acp_native.yaml`. Resource extraction checks ZIP
CRC and records each member's SHA256. The original checkpoint and receipts persist
in the results directory after deleting the runtime. Use `acp_native train` with
`--steps`, `--batch-size`, and optionally `--resume` for additional real-data updates.
The released force FFT uses 7000-Hz observations; a task with another force sampling
rate needs an explicit adapted sensor protocol.

This qualification does not reproduce the original hardware success table or
claim a full 300-epoch training run from the complete 230-episode dataset. The
physical controller lives in the author's separate C++ `force_control` and
`hardware_interfaces` repositories. Cross-EBDM task adapters and their physical
qualification are separate work. A state-only encoder is an adapted variant and
requires separate qualification and naming.

The explicit T3 state-history adaptation lives in
`genedynamics.learning.baselines.acp.task_training`. It uses public TCP, wrench,
twist, goal and elapsed-time histories, replaces the native image/force encoder
with a learned history encoder, and retains the original Mod1 U-Net, DDIM,
normalizer, EMA and structured 19D pose/virtual-target/stiffness actions.
`requirements/baselines/acp-tasks.txt` is the corresponding task profile.
`scripts.baselines.acp_tasks` trains from SHA-qualified source demonstrations,
checks exact continuation, and can stop with `--source-only`. The optional
`--rotation-residual` normalization is an explicit adaptation fitted entirely
from source rotations. It does not project predictions onto the goal.

Source controller diagnostics and workloads are reproducible through:

```bash
JAX_PLATFORMS=cpu .runtime/acp_native/venv/bin/python -m scripts.baselines.acp_gt_audit live --mode 0 --seed 18450 --output <fresh-source-results-directory>
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cpu .runtime/acp_native/venv/bin/python -m scripts.baselines.acp_tasks --source-only --rotation-residual --steps 2000 --seed <fresh-seed> --demonstration <mode0-episode.npz> --demonstration <mode1-episode.npz>
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cpu .runtime/acp_native/venv/bin/python -m scripts.baselines.acp_workloads native --output <fresh-native-workload-directory>
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cpu .runtime/acp_native/venv/bin/python -m scripts.baselines.acp_workloads adapted --adapted-checkpoint <policy_ema.pt> --output <fresh-source-workload-directory>
```

The live public expert encodes desired normal force as a virtual-target offset
and produces source demonstrations consistent with the actual ACP simulator
controller. Posthoc virtual-target labels from the different hybrid controller
can fail late even when the original demonstration succeeded. Both cases and
all physical replay costs are retained. Learned-policy source imitation and
target performance remain separate from controller correctness: lower diffusion
loss alone does not establish accurate rotations or successful contact behavior.
The workload entrypoint uses real source batches of sizes 1 and 2, actual encoders
and 16-step sampling; the adapted route additionally executes CPU Panda dynamics
and keeps every budget rejection and physical failure. Native workload timing
does not claim the original robot's execution latency or success rate.

The bounded source-precision workflow is configured in
`configs/arm/keyed_insertion_recovery/qualification/acp_source_precision.yaml`.
`scripts.baselines.acp_source_precision collect` retains every actual source
expert attempt and its charged physical replay; `train` splits complete episodes,
fits normalization on training episodes only, and checks fixed source sampling
gates. Balanced sampling and optimizer warmstarts have separate configurations
and result leaves. Every run verifies continuation through a fresh random
minibatch, including the model, EMA, optimizer, gradient, scheduler and RNG state.
These repeatedly used development validation episodes are not a final test set.

`scripts.baselines.acp_source_diagnostics` independently audits saved-source label
alignment, conditioning, clipping and DDIM step-count sensitivity.
`scripts.baselines.acp_source_wrench_audit` compares paired native anisotropic
K/VT expected wrenches at the same observed TCP. This static diagnostic excludes
damping, velocity feedback and plant dynamics. `scripts.baselines.acp_source_physical`
freezes an explicitly supplied final-budget EMA and predeclares a separate
four-episode Panda physical audit with exact charged replay. Its physical gate
does not overwrite an unsuccessful sampling gate; every physical failure remains
in the denominator. These entrypoints contain no target embodiment execution.

Local changes preserve native outputs in their regular domain: parameter-free
einops reshapes use Torch operations, and stiffness decoding uses an equivalent
orthogonal projector expression. The latter also avoids the author's singular
cross-product basis when the compliance direction is parallel to z. The virtual
target estimator keeps its numerical outputs while avoiding upstream mutation of
the caller's velocity vector. SE3 inversion preserves singleton batch/arm axes
when decoding batched policies. Optional data-only zarr imports are lazy; normalizer
fitting and checkpoint parameter structure are preserved.
