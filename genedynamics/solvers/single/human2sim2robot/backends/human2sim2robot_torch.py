"""Native Human2Sim2Robot actor deployment, independent of Isaac Gym.

Inference follows ``ppo/ppo_player.py`` at the commit in source_manifest.json.
Copyright (c) 2025 Tyler Lum. MIT License; see ../LICENSE.upstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .native.models import ModelA2CContinuousLogStd
from .native.network import MlpConfig, NetworkConfig, RnnConfig


def native_lstm_config() -> NetworkConfig:
    """The actor in upstream ``CrossEmbodimentPPOLSTM.yaml``."""
    return NetworkConfig(
        mlp=MlpConfig(units=[512, 512]),
        rnn=RnnConfig(
            units=1024,
            layers=1,
            name="lstm",
            layer_norm=True,
            before_mlp=False,
            concat_input=False,
            concat_output=True,
        ),
        separate_value_mlp=False,
    )


class Human2Sim2RobotBackendTorch:
    """Checkpoint-compatible native actor and episode-local recurrent state.

    ``dones`` passed to ``infer`` identifies episodes that ended *before* the
    current observation. It clears only those batch entries, as the native
    player's post-step reset does. Observations must be the public native task
    observation vector, in its trained order and units.
    """

    def __init__(
        self,
        *,
        network_config: NetworkConfig,
        observation_shape: Sequence[int],
        action_low: Any,
        action_high: Any,
        normalize_input: bool = True,
        normalize_value: bool = True,
        value_size: int = 1,
        clip_actions: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        if network_config.asymmetric_critic:
            raise ValueError("The deployed actor cannot use an asymmetric-critic-only network.")
        self.device = torch.device(device)
        self.observation_shape = tuple(observation_shape)
        if len(self.observation_shape) != 1 or self.observation_shape[0] <= 0:
            raise ValueError("Human2Sim2Robot expects a nonempty flat observation vector.")
        self.actions_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.actions_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        if self.actions_low.ndim != 1 or self.actions_low.numel() == 0:
            raise ValueError("Action bounds must be nonempty vectors.")
        if self.actions_high.shape != self.actions_low.shape:
            raise ValueError("Action bounds must have the same shape.")
        if not bool(torch.isfinite(self.actions_low).all() and torch.isfinite(self.actions_high).all()):
            raise ValueError("Action bounds must be finite.")
        if bool((self.actions_high < self.actions_low).any()):
            raise ValueError("Action high bounds must be at least action low bounds.")
        self.clip_actions = bool(clip_actions)
        self.model = ModelA2CContinuousLogStd(
            network_config=network_config,
            actions_num=self.actions_low.numel(),
            input_shape=self.observation_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            value_size=value_size,
            num_seqs=1,
        ).to(self.device)
        self.model.eval()
        self.states: tuple[torch.Tensor, ...] | None = None
        self.batch_size: int | None = None
        self.checkpoint_metadata: dict[str, Any] = {}

    def reset(self, batch_size: int | None = None, *, dones: Any = None) -> None:
        """Clear all history, or only ended episode slots in an existing batch."""
        if dones is not None:
            if batch_size is not None:
                raise ValueError("Use either batch_size or dones for reset.")
            if self.batch_size is None:
                raise ValueError("Cannot reset episode slots before initializing a batch.")
            mask = torch.as_tensor(dones, device=self.device, dtype=torch.bool).reshape(-1)
            if mask.numel() != self.batch_size:
                raise ValueError("Done mask must contain one entry per episode.")
            if self.states is not None:
                self.states = tuple(s.masked_fill(mask[None, :, None], 0.0) for s in self.states)
            return
        if batch_size is None:
            self.batch_size = None
            self.states = None
            return
        if int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.batch_size = int(batch_size)
        default = self.model.get_default_rnn_state()
        self.states = (
            tuple(torch.zeros((s.shape[0], self.batch_size, s.shape[2]),
                              dtype=torch.float32, device=self.device) for s in default)
            if default is not None else None
        )

    def restore(
        self,
        checkpoint: str | Path | Mapping[str, Any],
        *,
        weights_only: bool = True,
    ) -> None:
        """Load the native ``model`` and optional separate normalizer weights.

        File checkpoints use this backend's device, replacing upstream's
        hardcoded CUDA load. Optimizer and simulator state belong to the native
        training process and are not restored into this deployment solver.
        ``weights_only=False`` is available for trusted full training snapshots
        containing objects unsupported by PyTorch's restricted loader.
        """
        weights = (
            checkpoint if isinstance(checkpoint, Mapping)
            else torch.load(Path(checkpoint), map_location=self.device, weights_only=weights_only)
        )
        if not isinstance(weights, Mapping) or "model" not in weights:
            raise ValueError("Expected a native Human2Sim2Robot checkpoint containing 'model'.")
        self.model.load_state_dict(weights["model"], strict=True)
        if self.model.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"], strict=True)
        self.model.eval()
        self.checkpoint_metadata = {key: weights[key] for key in ("epoch", "frame") if key in weights}
        self.reset()

    @torch.no_grad()
    def infer(self, observation: Any, *, deterministic: bool = True, dones: Any = None) -> dict[str, Any]:
        """Run one native actor step; return executed action and model diagnostics."""
        if isinstance(observation, Mapping):
            if "obs" not in observation:
                raise ValueError("Observation mappings must contain the native flat 'obs' vector.")
            observation = observation["obs"]
        obs = torch.as_tensor(observation, device=self.device)
        unbatched = obs.ndim == 1
        if unbatched:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2 or obs.shape[1:] != self.observation_shape:
            raise ValueError(f"Expected observation shape [batch, {self.observation_shape[0]}].")
        obs = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs.float()
        batch_size = obs.shape[0]
        if self.batch_size is None:
            self.reset(batch_size)
        elif self.batch_size != batch_size:
            raise ValueError("Observation batch changed; call reset(batch_size) at the episode boundary.")
        if dones is not None:
            self.reset(dones=dones)
        result = self.model({
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        })
        self.states = result["rnn_states"]
        # The upstream model samples even when the player selects mu. Keep that
        # RNG consumption for same-seed deterministic/stochastic sequence parity.
        action = result["mus"] if deterministic else result["actions"]
        if self.clip_actions:
            action = torch.clamp(action, -1.0, 1.0)
            d = (self.actions_high - self.actions_low) / 2.0
            m = (self.actions_high + self.actions_low) / 2.0
            action = action * d + m
        result = dict(result)
        # Keep an action dimension even for a single scalar action. The native
        # player's unconditional squeeze drops it for unbatched one-action tasks.
        result["action"] = action[0].detach() if unbatched else action.detach()
        return result

    def act(self, observation: Any, *, deterministic: bool = True, dones: Any = None) -> torch.Tensor:
        return self.infer(observation, deterministic=deterministic, dones=dones)["action"]


__all__ = [
    "Human2Sim2RobotBackendTorch", "MlpConfig", "NetworkConfig", "RnnConfig", "native_lstm_config",
]
