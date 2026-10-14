"""Small-scale native IDM adaptation on public target action/observation records."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch

from genedynamics.solvers.single.fada.backends.fada_torch import FADABackendTorch
from genedynamics.solvers.single.fada.backends.lora import build_merged_state_dict
from .adaptation import configure_idm_finetune_method, _run_epoch
from .checkpoint import capture_rng_state, restore_rng_state, to_cpu


@dataclass(frozen=True)
class IDMAdaptationConfig:
    """Release recipe; scheduling/record collection is explicitly caller-owned."""

    finetune_method: str = "lora"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_scope: str = "encoder_decoder_qkv"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip: float = 0.5
    horizon_gamma: float = 0.0

    def __post_init__(self):
        import math

        values = (self.alpha, self.dropout, self.learning_rate, self.weight_decay, self.grad_clip, self.horizon_gamma)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("FADA adaptation hyperparameters must be finite")
        if self.rank <= 0 or self.alpha <= 0 or self.learning_rate <= 0:
            raise ValueError("rank, alpha and learning_rate must be positive")
        if not 0 <= self.dropout < 1 or not 0 <= self.horizon_gamma <= 1:
            raise ValueError("dropout must be in [0,1), horizon_gamma in [0,1]")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("weight_decay and grad_clip must be nonnegative")


class FADAIDMAdapter:
    """Adapt a loaded source student; never creates an oracle or consumes rewards.

    Target windows must come from complete trajectories through
    ``TrajectoryWindowDataset``; separate episodes before creating windows. Each
    ``step`` is one native AdamW update. Caller controls charged data and step budgets.
    """

    def __init__(self, checkpoint, *, device="cpu", config=None):
        self.config = config if config is not None else IDMAdaptationConfig()
        self._base_checkpoint = deepcopy(dict(checkpoint))
        self.backend = FADABackendTorch.from_checkpoint(checkpoint, device=device)
        self.model = self.backend.model
        cfg = self.config
        self.replaced_modules, self.method, self.target_scope = configure_idm_finetune_method(
            self.model, finetune_method=cfg.finetune_method, lora_r=cfg.rank,
            lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, lora_target_scope=cfg.target_scope,
        )
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        self.steps = 0

    def step(self, public_record_batch):
        cfg = self.config
        metrics = _run_epoch(
            model=self.model, loader=[public_record_batch], optimizer=self.optimizer,
            device=self.backend.device, io_norm_enabled=self.backend.io_normalization,
            obs_stats=self.backend.stats["obs"], action_stats=self.backend.stats["act"],
            command_stats=self.backend.stats["cmd"], pred_horizon_k=self.model.pred_horizon,
            grad_clip=cfg.grad_clip, show_progress=False, epoch_desc="FADA IDM adaptation",
            loss_target="idm", horizon_gamma=cfg.horizon_gamma,
        )
        self.steps += 1
        return metrics

    def export_checkpoint(self):
        """Return deployable native weights, preserving native normalization metadata."""
        payload = deepcopy(self._base_checkpoint)
        merged = {key: value.cpu() for key, value in build_merged_state_dict(self.model).items()}
        payload["model_state_dict"] = merged
        # Native loader prioritizes split weights, so replace both instead of retaining
        # stale source weights alongside the merged adaptation.
        payload["planner_state_dict"] = {key[len("planner."):]: value for key, value in merged.items() if key.startswith("planner.")}
        payload["idm_state_dict"] = {key[len("idm."):]: value for key, value in merged.items() if key.startswith("idm.")}
        payload["genedynamics_adaptation"] = {
            "method": self.method, "target_scope": self.target_scope, "optimizer_steps": self.steps,
            "source_commit": "290b3a9c5c0ebe2e9851b61350617544be05e9bc",
        }
        return payload

    def training_state_dict(self):
        """Snapshot optimizer, unmerged adapters, normalization and global RNG.

        Caller-owned DataLoader generators/cursors are intentionally external and must
        accompany this snapshot when resuming an input stream.
        """
        return {
            "schema": "genedynamics.fada.idm_training.v1", "config": asdict(self.config),
            "base_checkpoint": to_cpu(self._base_checkpoint),
            "model_state_dict": to_cpu(self.model.state_dict()),
            "optimizer_state_dict": to_cpu(self.optimizer.state_dict()),
            "steps": self.steps, "training": self.model.training, "rng": capture_rng_state(),
        }

    @classmethod
    def from_training_state(cls, state, *, device="cpu", restore_rng=True):
        if state.get("schema") != "genedynamics.fada.idm_training.v1":
            raise ValueError("Unsupported FADA adaptation resume schema")
        instance = cls(state["base_checkpoint"], device=device, config=IDMAdaptationConfig(**state["config"]))
        instance.model.load_state_dict(state["model_state_dict"], strict=True)
        instance.optimizer.load_state_dict(state["optimizer_state_dict"])
        instance.steps = int(state["steps"])
        instance.model.train(bool(state["training"]))
        if restore_rng:
            restore_rng_state(state["rng"])
        return instance
