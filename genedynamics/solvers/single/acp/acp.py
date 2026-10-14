"""Closed-loop ACP deployment preserving native pose/virtual-target/stiffness semantics."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .compliance import decode_action19, decode_stiffness_tool
from .policy import ACPDiffusionPolicy


class ACPSolver:
    """Deploy the native sparse ACP diffusion model using a task-owned controller.

    The public observation contains ``sparse`` native, prepared sensor histories
    (B, T_key, ...), and ``current_tool_pose_world`` (B, R, 4, 4). Histories are
    *not* normalized yet: the checkpoint's sparse_normalizer performs this step.
    RGB is float [0, 1] in B,T,C,H,W; pose histories use the author's relative
    row-based rotation6d; wrench is (Fx,Fy,Fz,Tx,Ty,Tz), N/Nm, in the native
    observation frame. The task adapter owns sampling frequencies, sensor frames
    and construction of these public histories.

    Output contains world reference and virtual-target poses, current-tool-frame
    stiffness matrices and waypoint times in seconds. It does not flatten these
    controls into another method's action representation. Train via the actual
    ``policy.compute_loss(batch, args)`` outside ``act``/``predict_action``.
    """

    def __init__(
        self,
        policy: ACPDiffusionPolicy,
        shape_meta: Mapping[str, Any],
        *,
        device: str = "cpu",
        raw_sample_interval_s: float = 0.002,
        slow_down_factor: float = 1.5,
        execution_horizon: int = 12,
    ):
        if not isinstance(policy, ACPDiffusionPolicy):
            raise TypeError("ACP requires its source-derived DiffusionUnetTimmMod1Policy")
        action_dim = int(shape_meta["action"]["shape"][0])
        if action_dim not in (19, 38) or policy.action_dim != action_dim:
            raise ValueError("ACP requires native 19D or 38D pose/virtual-target/stiffness actions")
        action_meta = shape_meta["sample"]["action"]["sparse"]
        if (int(action_meta["horizon"]) != policy.sparse_action_horizon
                or int(action_meta["down_sample_steps"]) != policy.sparse_action_down_sample_steps):
            raise ValueError("ACP policy and deployment shape_meta disagree about waypoint sampling")
        if not np.isfinite(raw_sample_interval_s) or raw_sample_interval_s <= 0:
            raise ValueError("raw_sample_interval_s must be finite and positive")
        if not np.isfinite(slow_down_factor) or slow_down_factor <= 0:
            raise ValueError("slow_down_factor must be finite and positive")
        if not 1 <= execution_horizon <= policy.sparse_action_horizon:
            raise ValueError("execution_horizon must be within the predicted sparse action horizon")
        self.policy = policy.to(torch.device(device)).eval()
        self.shape_meta = shape_meta
        self.device = torch.device(device)
        self.execution_horizon = int(execution_horizon)
        self.sample_interval_s = (
            float(raw_sample_interval_s) * float(slow_down_factor)
            * int(policy.sparse_action_down_sample_steps)
        )

    def reset(self):
        self.policy.reset()
        self.policy.sparse_nobs_encode = None
        self.policy.sparse_naction_pred = None

    def _tensor_observations(self, sparse):
        if not isinstance(sparse, Mapping):
            raise TypeError("sparse observation must be a mapping of public sensor histories")
        result = {}
        batch_size = None
        for key, config in self.shape_meta["sample"]["obs"]["sparse"].items():
            if key not in sparse:
                raise KeyError(f"missing ACP public observation history: {key}")
            value = torch.as_tensor(sparse[key], device=self.device, dtype=self.policy.dtype)
            expected_shape = tuple(self.shape_meta["obs"][key]["shape"])
            if value.ndim != 2 + len(expected_shape) or tuple(value.shape[2:]) != expected_shape:
                raise ValueError(f"{key} must have shape (B, T, {expected_shape})")
            if value.shape[1] != int(config["horizon"]):
                raise ValueError(f"{key} history length differs from native shape_meta")
            if not torch.isfinite(value).all():
                raise ValueError(f"{key} contains non-finite sensor values")
            if batch_size is not None and batch_size != value.shape[0]:
                raise ValueError("ACP public sensor histories must have the same batch size")
            batch_size = value.shape[0]
            result[key] = value
        if not result:
            raise ValueError("ACP requires the native sensor history schema")
        return result

    @torch.inference_mode()
    def predict_action(self, sparse, current_tool_pose_world):
        observations = self._tensor_observations(sparse)
        raw_action = self.policy.predict_action({"sparse": observations})["sparse"]
        if torch.is_tensor(current_tool_pose_world):
            current_tool_pose_world = current_tool_pose_world.detach().cpu().numpy()
        current = np.asarray(current_tool_pose_world)
        reference, virtual, scalar = decode_action19(raw_action.detach().cpu().numpy(), current)
        stiffness = decode_stiffness_tool(reference, virtual, scalar, current[:, None])
        return {
            "reference_pose_world": reference,
            "virtual_target_pose_world": virtual,
            "stiffness_tool": stiffness,
            "stiffness_scalar": scalar,
            "waypoint_times_s": np.arange(raw_action.shape[1]) * self.sample_interval_s,
            "execution_horizon": self.execution_horizon,
            "raw_action": raw_action,
        }

    def act(self, public_observation):
        if not isinstance(public_observation, Mapping):
            raise TypeError("ACP requires a task-owned public observation mapping")
        return self.predict_action(
            public_observation["sparse"], public_observation["current_tool_pose_world"]
        )

    def load_policy_state_dict(self, payload, *, use_ema: bool = True):
        """Strictly load native workspace ``state_dicts`` or an explicit state dict.

        Selecting EMA never silently falls back to the non-EMA model. Downloaded
        author's dill/Hydra workspace checkpoints must first be loaded in their
        native environment and their tensor state_dict exported; this method
        does not replace the original encoder or unpickle arbitrary workspace data.
        """
        if "state_dicts" in payload:
            key = "ema_model" if use_ema else "model"
            if key not in payload["state_dicts"]:
                raise KeyError(f"native ACP checkpoint has no {key}")
            state_dict = payload["state_dicts"][key]
        else:
            state_dict = payload
        self.policy.load_state_dict(state_dict, strict=True)
        self.policy.to(self.device).eval()
        self.reset()

    def load_checkpoint(self, path: str | Path, *, use_ema: bool = True):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.load_policy_state_dict(payload, use_ema=use_ema)


__all__ = ["ACPSolver"]
