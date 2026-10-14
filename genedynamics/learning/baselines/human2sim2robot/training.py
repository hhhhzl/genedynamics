"""Run the actual pinned H2S2R PPO learner and retain complete continuation state.

The native learner remains in its reference checkout. Task environments must
declare their adaptations separately; this is not a substitute for human data,
digital-twin construction or the original FABRICS/IsaacGym controller.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "baselines/Human2Sim2Robot"


def source_learner():
    from scripts.baselines.sources import LOCK, verify
    import json

    spec = next(s for s in json.loads(LOCK.read_text())["sources"] if s["method"] == "human2sim2robot")
    verify(spec)
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from human2sim2robot.ppo.ppo_agent import PpoAgent, PpoConfig
    from human2sim2robot.ppo.utils.network import MlpConfig, NetworkConfig, RnnConfig
    from human2sim2robot.ppo.utils.rewards_shaper import RewardsShaperParams
    return PpoAgent, PpoConfig, NetworkConfig, MlpConfig, RnnConfig, RewardsShaperParams


def build_learner(env, output, *, device="cuda:0", num_actors=2, horizon=16,
                  mlp_units=(512, 512), rnn_units=1024, epochs=2000000):
    Agent, Config, Network, Mlp, Rnn, Rewards = source_learner()
    network = Network(
        mlp=Mlp(units=list(mlp_units)),
        rnn=Rnn(name="lstm", units=rnn_units, layers=1, before_mlp=False,
                layer_norm=True, concat_input=False, concat_output=True),
        separate_value_mlp=False,
    )
    # Native release settings; batch count/epoch budget reflect this workload.
    config = Config(
        num_actors=num_actors, learning_rate=1e-4, entropy_coef=0.,
        horizon_length=horizon, normalize_advantage=True, normalize_input=True,
        grad_norm=1., critic_coef=4., gamma=.998, tau=.95,
        reward_shaper=Rewards(scale_value=1.), mini_epochs=4, e_clip=.2,
        device=device, normalize_value=True, truncate_grads=True,
        seq_length=horizon, minibatch_size=num_actors * horizon,
        bound_loss_type="regularisation", bounds_loss_coef=.005,
        zero_rnn_on_done=True, lr_schedule="linear", schedule_type="standard",
        max_epochs=epochs, print_stats=False,
    )
    agent = Agent(Path(output), config, network, env)
    agent.init_tensors()
    agent.reset_envs()
    return agent, network


_CONTINUATION_FIELDS = (
    "obs", "rnn_states", "dones", "current_rewards", "current_shaped_rewards",
    "current_lengths", "current_lr", "current_entropy_coef", "is_tensor_obses",
    "mean_rewards", "last_mean_rewards", "frame", "epoch_num", "curr_frames",
)


def capture_training_state(agent):
    """Add RNG, rollout memory and scheduler state omitted by native snapshots."""
    numpy_rng = np.random.get_state()
    cuda_rng = None
    if torch.device(agent.device).type == "cuda":
        cuda_rng = torch.cuda.get_rng_state(torch.device(agent.device))
    return deepcopy({
        "schema": "h2s2r-training-continuation-v1",
        "native": agent.get_full_state_weights(),
        "continuation": {name: getattr(agent, name) for name in _CONTINUATION_FIELDS},
        "scheduler": agent.scheduler.__dict__,
        "meters": {name: {"state": getattr(agent, name).state_dict(),
                           "current_size": int(getattr(agent, name).current_size)}
                   for name in ("game_rewards", "game_shaped_rewards", "game_lengths")},
        "rng": {"python": random.getstate(),
                "numpy": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                "torch_cpu": torch.get_rng_state(), "torch_cuda": cuda_rng},
    })


def restore_training_state(agent, payload):
    if payload["schema"] != "h2s2r-training-continuation-v1":
        raise ValueError("unexpected training state schema")
    agent.set_full_state_weights(payload["native"])
    def on_device(value):
        if torch.is_tensor(value):
            return value.to(agent.device)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(on_device(item) for item in value)
        return value
    for name, value in deepcopy(payload["continuation"]).items():
        setattr(agent, name, on_device(value))
    for name, meter in payload["meters"].items():
        getattr(agent, name).load_state_dict(meter["state"])
        getattr(agent, name).current_size = int(meter["current_size"])
    agent.scheduler.__dict__.clear()
    agent.scheduler.__dict__.update(deepcopy(payload["scheduler"]))
    for group in agent.optimizer.param_groups:
        group["lr"] = agent.current_lr
    rng = payload["rng"]
    random.setstate(rng["python"])
    numpy_rng = rng["numpy"]
    np.random.set_state((numpy_rng[0], np.array(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if rng["torch_cuda"] is not None:
        if torch.device(agent.device).type != "cuda":
            raise ValueError("exact CUDA training continuation requires CUDA")
        torch.cuda.set_rng_state(rng["torch_cuda"].cpu(), torch.device(agent.device))


def train_epoch(agent):
    agent.update_epoch()
    result = agent.train_epoch()
    agent.frame += agent.curr_frames
    # The native outer loop clears the PPO dataset between epochs.
    agent.dataset.update_values_dict(None)
    return {
        "actor_loss": [float(x.detach().cpu()) for x in result[4]],
        "critic_loss": [float(x.detach().cpu()) for x in result[5]],
        "entropy": [float(x.detach().cpu()) for x in result[7]],
        "kl": [float(x.detach().cpu()) for x in result[8]],
        "learning_rate": float(agent.current_lr), "epoch": int(agent.epoch_num),
        "frame": int(agent.frame), "play_seconds": float(result[1]),
        "update_seconds": float(result[2]),
    }
