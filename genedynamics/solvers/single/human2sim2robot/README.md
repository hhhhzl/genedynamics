# Human2Sim2Robot native actor solver

This ports the native PyTorch PPO model, input/value normalization, shared or
separate actor/critic MLPs, and LSTM/GRU recurrence from
[`tylerlum/human2sim2robot`](https://github.com/tylerlum/human2sim2robot)
commit `894eae2ec3ae39a573b81bd1860d14cc6bdfa6df`. Parameter names are unchanged,
so native `checkpoint["model"]` weights load strictly. The original MIT license
is preserved in `LICENSE.upstream`; `source_manifest.json` records source hashes.

```python
from genedynamics.solvers.single.human2sim2robot import (
    Human2Sim2RobotSolver, native_lstm_config,
)

solver = Human2Sim2RobotSolver(
    network_config=native_lstm_config(),
    observation_shape=(native_observation_dim,),
    action_low=native_action_space.low,
    action_high=native_action_space.high,
    device="cuda:0",
)
solver.restore("native_checkpoint.pth")
solver.reset(batch_size=num_envs)
action = solver.act(public_native_observation, deterministic=True)
# After env.step returns done, before the next observation:
solver.reset(dones=done)
```

`infer` also returns native `mus`, `sigmas`, `values`, log probabilities, sampled
actions and recurrent state. `action` is the selected executed action after the
native clamp-to-[-1,1] and affine action-space rescaling. With `clip_actions=False`
the original unscaled policy action is returned. As upstream does, deterministic
inference still consumes the model's Gaussian sample; same RNG states therefore
produce matching later stochastic actions. Normalizer statistics remain frozen
in deployment. A batch size change requires an explicit episode reset.

This actor integration is only one stage of Human2Sim2Robot. Native training
stays in the reference checkout: `human2sim2robot/sim_training/run.py`, the native
PPO learner and the video-derived object trajectory reward, robot/environment
digital twins and demonstration-based initialization must all be retained before
claiming reproduction of the full method. A replacement arm, observation vector,
controller or demonstration protocol is an adapted variant and needs independent
qualification. Actor dimensions or matching network weights alone cannot provide
that qualification.

The pinned upstream installation requires Isaac Gym Preview 4 (Python 3.7/3.8),
PyTorch 2.1.2+cu118, the Human2Sim2Robot FABRICS branch at
`b56c9b1d7c927f5f8b7ca9464a3b5607a3777dae`, and a cuRobo fork. Its assets,
processed/raw demonstrations and pretrained models require the author download
URL supplied through the form in the upstream README. No assets, trained weights,
or native GPU training results are supplied by this actor port. The solver itself
requires only PyTorch and the repository's usual package dependencies.

The independent parity suite imports native source from `baselines/Human2Sim2Robot`
and compares native checkpoint restore, normalization, gradients, MLP/LSTM/GRU
outputs, partial episode resets, clipped/scaled actions and RNG consumption:

```sh
.runtime/cross_ebdm/venv/bin/python -m pytest -q \
  test/baselines/test_human2sim2robot_source_parity.py
```

Set `GENEDYNAMICS_BASELINE_TEST_DEVICE=cuda:0` and use a CUDA PyTorch runtime
to run the same reference comparisons on GPU.

Documented integration differences: file checkpoints map to the requested device
instead of upstream's hardcoded CUDA loader; unbatched single-action output keeps
its action dimension; unsupported batch changes and invalid shapes fail explicitly;
native optimizer/simulator state is not restored into the deployment solver.

The separate `genedynamics.learning.baselines.human2sim2robot` training bridge
now calls the original upstream `PpoAgent` and preserves its task reward/reset
math. `scripts.baselines.human2sim2robot_tasks` trains on a source robot object
trajectory in a public nominal target-body model; it supports real CPU physics,
complete optimizer/RNG/RNN/normalizer/physics continuation, charged target
episodes, exact CPU replay and batch1/4 end-to-end workload profiles. This is an
explicit held-object/state/shared-controller adaptation. It does not supply the
gated human video/assets or certify native FABRICS/IsaacGym performance.

Use `scripts.baselines.environment run --name baseline_tasks_torch --profile
tasks-pytorch --device cuda -- python -m scripts.baselines.human2sim2robot_tasks
--help` from the repository root. The wrapper keeps caches inside the owned
runtime. `--epochs` controls the actual native PPO training budget. Current
implementation and qualification evidence are documented in
[the P3 execution record](../../../../docs/development/cross-embodiment-p3-execution.md).
