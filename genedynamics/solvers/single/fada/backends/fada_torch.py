"""Native execution adapter around the source-derived FADA Planner-IDM.

The task adapter supplies public histories. This module never reads simulator state.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from typing import Any, Mapping

import torch

from .lora import LoRALinear, LoRAMultiheadAttention, _extract_norm_stats
from .model import PlannerIDMPolicy

_MHA_LOCK = RLock()


@contextmanager
def lora_inference_context(model):
    """Preserve adapters in Transformer evaluation, matching upstream's guard."""
    has_lora = any(isinstance(m, (LoRALinear, LoRAMultiheadAttention)) for m in model.modules())
    if not has_lora:
        yield
        return
    # PyTorch's encoder fastpath directly reads frozen base weights, bypassing LoRA.
    with _MHA_LOCK:
        previous = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            yield
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)


class FADABackendTorch:
    def __init__(
        self,
        model: PlannerIDMPolicy,
        *,
        device: str = "cpu",
        io_normalization: bool,
        obs_norm_stats: Mapping[str, Any] | None = None,
        action_norm_stats: Mapping[str, Any] | None = None,
        command_norm_stats: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        self.device = torch.device("cuda" if device == "gpu" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested FADA device {self.device}, but CUDA is unavailable")
        if not isinstance(model, PlannerIDMPolicy):
            raise TypeError("FADA requires its source-derived PlannerIDMPolicy")
        self.model = model.to(device=self.device).eval()
        self.io_normalization = bool(io_normalization)
        self.metadata = deepcopy(dict(metadata or {}))
        self.stats = {}
        definitions = (
            ("obs", "obs_norm_stats", obs_norm_stats, "obs_mean", "obs_std", model.obs_dim),
            ("act", "action_norm_stats", action_norm_stats, "action_mean", "action_std", model.act_dim),
            ("cmd", "command_norm_stats", command_norm_stats, "command_mean", "command_std", model.cmd_dim),
        )
        for name, field, raw, mean_key, std_key, dim in definitions:
            stats = _extract_norm_stats(
                {field: dict(raw) if raw is not None else None}, key=field,
                mean_key=mean_key, std_key=std_key, dim=dim,
                required=self.io_normalization, device=self.device,
            )
            self.stats[name] = stats

    def predict(self, history_obs, history_act, current_command, *, history_valid_mask, return_obs=False):
        model = self.model
        dtype = next(model.parameters()).dtype
        obs = torch.as_tensor(history_obs, device=self.device, dtype=dtype)
        act = torch.as_tensor(history_act, device=self.device, dtype=dtype)
        cmd = torch.as_tensor(current_command, device=self.device, dtype=dtype)
        mask = torch.as_tensor(history_valid_mask, device=self.device)
        if mask.dtype != torch.bool:
            mask = mask > 0
        if obs.ndim != 3 or mask.shape != obs.shape[:2]:
            raise ValueError("FADA requires batched history [B,H,D] and validity mask [B,H]")
        if obs.shape[1] != model.history_len:
            raise ValueError("FADA public history length differs from its checkpoint")
        if not bool(mask[:, -1].all()):
            raise ValueError("FADA histories must be right aligned with a valid most recent observation")
        if not all(bool(torch.isfinite(value).all()) for value in (obs, act, cmd)):
            raise ValueError("FADA public history/command contains non-finite values")
        if self.io_normalization:
            obs = (obs - self.stats["obs"]["mean"].view(1, 1, -1)) / self.stats["obs"]["std"].view(1, 1, -1)
            act = (act - self.stats["act"]["mean"].view(1, 1, -1)) / self.stats["act"]["std"].view(1, 1, -1)
            cmd = (cmd - self.stats["cmd"]["mean"].view(1, -1)) / self.stats["cmd"]["std"].view(1, -1)
        model.eval()
        with torch.inference_mode(), lora_inference_context(model):
            actions, future = model(obs, act, cmd, history_valid_mask=mask, return_obs=True)
        if self.io_normalization:
            actions = actions * self.stats["act"]["std"].view(1, 1, -1) + self.stats["act"]["mean"].view(1, 1, -1)
            future = future * self.stats["obs"]["std"].view(1, 1, -1) + self.stats["obs"]["mean"].view(1, 1, -1)
        return (actions, future) if return_obs else actions

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any], *, device="cpu", metadata=None):
        """Load native merged or split weights with checkpoint architecture and stats.

        Adapter-only payloads require their original base checkpoint and are deliberately
        not accepted here. ``build_merged_state_dict`` creates deployable native weights.
        """
        cfg = payload["cfg"]
        if not isinstance(cfg, Mapping):
            raise TypeError("FADA checkpoint cfg must be a mapping")
        required = (
            "history_len", "pred_horizon", "planner_d_model", "planner_nhead",
            "planner_num_layers", "planner_dim_feedforward", "planner_dropout",
            "planner_use_learned_positional_encoding", "idm_d_model", "idm_nhead",
            "idm_dim_feedforward", "idm_dropout", "idm_use_learned_positional_encoding",
        )
        params = {name: cfg[name] for name in required}
        params.update({name: int(payload[name]) for name in ("obs_dim", "act_dim", "cmd_dim")})
        params.update(
            idm_encoder_num_layers=int(cfg.get("idm_encoder_num_layers", cfg.get("idm_num_layers", 6))),
            idm_decoder_num_layers=int(cfg.get("idm_decoder_num_layers", cfg.get("idm_num_layers", 6))),
            planner_use_action_history=bool(cfg.get("planner_use_action_history", False)),
            idm_use_current_command_for_history=bool(cfg.get("idm_use_current_command_for_history", False)),
            planner_predict_delta=bool(cfg.get("planner_predict_delta", False)),
        )
        # Match native evaluator's preservation of rollout RNG while initializing
        # layers whose values will immediately be replaced by checkpoint weights.
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            model = PlannerIDMPolicy(**params)
        planner, idm = payload.get("planner_state_dict"), payload.get("idm_state_dict")
        if isinstance(planner, Mapping) and isinstance(idm, Mapping):
            model.load_split_state_dict(planner_state_dict=planner, idm_state_dict=idm)
        else:
            model.load_compatible_state_dict(payload["model_state_dict"])
        # The released native evaluator reads this top-level field, not cfg.
        normalization = payload.get("normalization")
        io_enabled = bool(normalization.get("io_enabled", False)) if isinstance(normalization, Mapping) else False
        return cls(
            model, device=device, io_normalization=io_enabled,
            obs_norm_stats=payload.get("obs_norm_stats"),
            action_norm_stats=payload.get("action_norm_stats"),
            command_norm_stats=payload.get("command_norm_stats"), metadata=metadata,
        )
