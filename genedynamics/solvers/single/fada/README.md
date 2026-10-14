FADA's native PyTorch Planner-IDM is integrated as `FADASolver`. The Transformer
planner, Transformer inverse-dynamics model, packed QKV and linear LoRA, target
trajectory windows and inverse-dynamics action loss are derived from the pinned
official source. [source_manifest.json](source_manifest.json) records source paths,
symbols and SHA-256 hashes. The copied code is covered by [LICENSE](LICENSE) and
[NOTICE](NOTICE).

```python
from genedynamics.solvers.single.fada import FADASolver
from genedynamics.learning.baselines.fada import FADAIDMAdapter

# payload is a native checkpoint dictionary, loaded by the caller.
policy = FADASolver.from_checkpoint(payload, device="cuda:0")
action = policy.act({
    "history_obs": history_obs,          # [B,H,D], raw public observations
    "history_act": history_act,          # [B,H,A], action preceding each observation
    "current_command": command,         # [B,C], current public task command
    "history_valid_mask": valid_mask,    # [B,H], right-aligned history
})

# Target records contain observations/actions/commands, without reward or hidden truth.
adapter = FADAIDMAdapter(payload, device="cuda:0")
metrics = adapter.step(target_record_batch)
adapted_policy = FADASolver.from_checkpoint(adapter.export_checkpoint(), device="cuda:0")
```

Native checkpoint normalization is controlled by
`payload["normalization"]["io_enabled"]`; means, standard deviations and their
floors are validated before use. The same source normalization maps are used for
inference and target adaptation. `predict(..., return_obs=True)` returns the full
action and planned-observation horizons in physical coordinates. `act` defaults to
one action per inference; `open_loop_deploy_steps` permits a bounded prefix, and
`reset` discards cached actions at episode or batch reset. The task owns the public
observation adapter, robot command units, frequency and episode partitioning.

`FADAIDMAdapter` freezes the planner, uses the released IDM-only LoRA/AdamW recipe,
and exports merged or split native weights without leaving stale source weights
in a higher-priority checkpoint field. The caller supplies sampling, charged data
and optimizer-step budgets; the released small-scale recipe uses 100 updates and
batches of 4096, which are not silently selected by the policy API.

This policy module covers source and numerical integration of the student and
adaptation core. The separate native entrypoints below execute author-checkpoint
oracle control/PPO, DAgger and target-record adaptation on the official G1 mjwarp
route. Execution receipts report learner resumption and closed-loop failures
separately. Native paper performance and the four Cross-EBDM task qualifications
require their own performance gates.
Existing upstream behavior is preserved, including the detached merged
`out_proj` path inside the LoRA attention wrapper; changing that gradient behavior
would require a separately identified implementation variant.

On Torch 2.5.1 CUDA, the native merged-checkpoint Transformer fastpath and the
unmerged LoRA evaluation path can differ by approximately `2e-4` in the small
qualification case. The same difference occurs upstream. Parity therefore checks
the port against upstream under the same execution mode exactly, and separately
checks adapter merging with the fastpath disabled for both models. No tolerance
is relaxed to hide an implementation discrepancy.

With the official clone present, run source parity on CPU or CUDA:

```bash
python -m pytest -q test/baselines/test_fada_source_parity.py
GENEDYNAMICS_BASELINE_TEST_DEVICE=cuda:0 python -m pytest -q test/baselines/test_fada_source_parity.py
```

The suite executes pinned upstream definitions independently of the port and
checks outputs, input gradients, target-window alignment, LoRA updates, frozen
planner weights, native normalization, merged checkpoints and serialized reload.

Native robot and training execution now have separate runnable entrypoints:

```bash
python -m scripts.baselines.fada_native_environment create --reuse-python .runtime/baseline_core_torch/venv/bin/python
python -m scripts.baselines.fada_native_assets fetch --robot g1 --selection history --output .runtime/fada_native/assets
```

`scripts.baselines.fada_native` provides `oracle`, `dagger`, `student`, `collect`
and `adapt` stages. Each stage requires an explicit config-mirrored `--output`;
the native configuration is
`configs/locomotion/g1_velocity_tracking/qualification/fada_native.yaml`.
`--profile qualification` keeps the released model, weak-policy data, strict
oracle-shadow labels and separate planner/IDM optimizers while bounding collection
and update budgets. `--profile release` retains the source learner budgets;
parallel-environment count remains an explicit resource choice. `collect` charges
the sum of target action seconds across environments, defaults to 120 seconds,
and writes native H5 episodes containing only public dynamics observations,
commands, executed actions and done boundaries.

The official mjwarp backend needs a small explicit compatibility adapter for
`MjwDofStateView.detach()` during DAgger snapshots. The wrapper installs the tensor
read adapter in both the parent and spawned oracle-shadow worker. Original source
files remain unchanged. The route uses MuJoCo 3.6.0, Warp 1.10.0 and the author-pinned
MuJoCo Warp commit `09ec1da65ac44bd68703f9507c9c9e8f22ae7c71`; its simulation differs
from the released IsaacSim source-training preset and therefore has its own
qualification label. Unsupported randomizers are never silently ignored.

`FADAIDMAdapter.training_state_dict()` and `from_training_state()` preserve unmerged
adapters, optimizer moments, normalization and Python/NumPy/CPU/CUDA RNG. Native
adaptation additionally saves the record permutation and cursor. Its next-update
resume check executes real recorded target batches, and exports the budgeted
snapshot before those verification updates for subsequent evaluation.

For target qualification, `--num-envs 1 --fixed-target-dynamics --startup-seed 42`
holds one sampled mass/COM/friction/PD/delay robot across resets; `--seed` controls
independent reset posture and velocity commands. Reset posture, pushes and torque
disturbances remain active. This evaluator-only protocol does not alter source
randomization or supply target parameters to the policy/fitter. Pre/post receipts
must have identical coefficient hashes and command schedule hashes. Source
equivalence auditing compares the port with the original model on actual native
histories, including native observation scaling, commands and normalization.

Native `reset_buf` is integer-valued. The runner converts it to a boolean mask
before clearing each environment's public history. The earlier qualification
receipts affected by integer indexing are retained with an explicit supersession
receipt and charged target data; they cannot qualify closed-loop performance.

Packages and caches are removable with
`python -m scripts.baselines.fada_native_environment clean`; source references,
asset SHA receipts and selected native checkpoints live separately in results.
Execution and save/restore qualification does not establish convergence or native
paper performance. Those gates require actual closed-loop stability and matched
pre/post results, including failed trials.
