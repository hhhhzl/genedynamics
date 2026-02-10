from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class SafeDiffuserPlanResult:
    states: list[np.ndarray]
    actions: list[np.ndarray]
    info: Dict[str, Any]


class SafeDiffuserBackendTorch:
    """
    Torch backend that matches DPCC's denoising architecture (third_party/diffuser),
    but applies SafeDiffuser-style safety correction on every denoising step via
    `projector.invariance(x_prev, xp1)`.
    """

    def __init__(
        self,
        *,
        env: Any,
        diffusion: Any,
        normalizer: Any,
        plan_config: Dict[str, Any],
        device: str = "cuda",
        seed: int = 0,
        goal_xy: Optional[np.ndarray] = None,
    ):
        self.env = env
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.plan_config = plan_config
        self.device = device
        self.seed = int(seed)
        self.goal_xy = None if goal_xy is None else np.asarray(goal_xy, dtype=np.float32).reshape(2)

        # SafeDiffuser execution details.
        # For avoiding-d3il, we typically constrain/correct the XY *position* components in obs.
        # Default (0,1) corresponds to obs = [x_des, y_des, x, y] (desired position components).
        self.pos_idx: Tuple[int, int] = tuple(self.plan_config.get('safediffuser_pos_idx', (0, 1)))  # type: ignore
        self.derive_action_from_states: bool = bool(self.plan_config.get('derive_action_from_states', True))

        self._policy = None
        self._projector = None
        self._stepper = None

    def _ensure_ready(self) -> None:
        if self._policy is not None:
            return
        from enerdynamics.solvers.single.safediffuser.patch.avoiding_cbf_qp import (
            AvoidingCBFConfig,
            AvoidingCBFQPCorrector,
        )
        from enerdynamics.solvers.single.safediffuser.stepper import SafeDiffuserTorchStepper
        from enerdynamics.solvers.single.safediffuser.policies import SafeDiffuserPolicy

        diffusion = self.diffusion
        normalizer = self.normalizer

        # Route denoising loop through solver-side SafeDiffuser stepper (DPCC-like).
        if self._stepper is None:
            self._stepper = SafeDiffuserTorchStepper(diffusion)
            diffusion.p_sample = self._stepper.p_sample
            diffusion.p_sample_loop = self._stepper.p_sample_loop

        projector = None

        enable_safety = bool(self.plan_config.get("enable_safety", True))
        if enable_safety:
            cfg_path = str(self.plan_config.get("safediffuser_config_path", "enerdynamics/solvers/single/safediffuser/config/avoiding_d3il.yaml"))
            correct_all_steps = bool(self.plan_config.get("correct_all_steps", True))
            projector = AvoidingCBFQPCorrector(
                normalizer=normalizer,
                config=AvoidingCBFConfig(
                    constraint_config_path=cfg_path,
                    exp=str(self.plan_config.get("exp", "avoiding-d3il")),
                    des_idx=tuple(self.pos_idx),
                    correct_all_steps=correct_all_steps,
                ),
            )

        self._projector = projector
        # Use a SafeDiffuser-specific policy wrapper (no trajectory_selection semantics).
        self._policy = SafeDiffuserPolicy(
            model=diffusion,
            normalizer=normalizer,
            projector=projector,
            preprocess_fns=self.plan_config.get("preprocess_fns", []),
            test_ret=float(self.plan_config.get("test_ret", 0)),
        )
        self.which_trajectory=int(self.plan_config.get("which_trajectory", 0))

    def _build_goal_state_4d(self) -> np.ndarray:
        """
        Build a 4D goal observation aligned with D3ILAvoiding state:
          [x_des, y_des, x, y]
        """
        goal_xy = self.goal_xy
        if goal_xy is None:
            goal_xy = np.asarray(getattr(self.env, "target"), dtype=np.float32).reshape(-1)[:2]
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        return np.array([gx, gy, gx, gy], dtype=np.float32)

    def plan(self, x0: np.ndarray, *, rng_key: Any | None = None) -> Dict[str, Any]:
        """
        Plan a full horizon trajectory.

        Args:
            x0: initial state, expected shape (4,) for D3ILAvoiding: [x_des, y_des, x, y]
            rng_key: optional RNG (used to seed torch if possible)
        """
        x0 = np.asarray(x0, dtype=np.float32).reshape(-1)
        if x0.size != 4:
            raise ValueError(f"SafeDiffuser backend expects x0 shape (4,), got {x0.shape}")

        self._ensure_ready()

        # Seeding (best-effort): make sampling deterministic per call when desired.
        try:
            if rng_key is not None:
                seed = int(getattr(rng_key, "integers", lambda low, high: 0)(0, 2**31 - 1))
            else:
                seed = self.seed
            torch.manual_seed(seed)
        except Exception:
            pass

        cond: Dict[int, np.ndarray] = {}
        horizon = int(self.plan_config.get("horizon", getattr(self.diffusion, "horizon", 8)))
        
        x0 = x0.copy()
        cond[0] = x0
        cond[horizon - 1] = self._build_goal_state_4d()

        batch_size = int(self.plan_config.get("batch_size", 4))
        all_sampled_action, trajectories, diffusion_paths = self._policy(
            cond,
            batch_size=batch_size,
            horizon=horizon,
            return_diffusion=bool(self.plan_config.get("return_diffusion", True)),
        )

        # third_party Policy returns: trajectories.actions [B,H,A], trajectories.observations [B,H,O]
        actions_raw = np.asarray(trajectories.actions[self.which_trajectory], dtype=np.float32)
        states = np.asarray(trajectories.observations[self.which_trajectory], dtype=np.float32)

        # If the projector modifies the *state* samples (SafeDiffuser invariance hook), the
        # action slice produced by the diffusion model can become inconsistent. In SafeDiffuser
        # implementations that execute controls derived from the generated (corrected) states,
        # we should compute actions from the corrected states.
        actions_exec = actions_raw
        if (
            self.derive_action_from_states
            and states.ndim == 2
            and states.shape[0] >= 2
            and actions_raw.shape[-1] == 2
        ):
            i0, i1 = int(self.pos_idx[0]), int(self.pos_idx[1])
            if 0 <= i0 < states.shape[1] and 0 <= i1 < states.shape[1]:
                pos = states[:, [i0, i1]]
                delta = pos[1:] - pos[:-1]
                actions_exec = np.zeros_like(actions_raw, dtype=np.float32)
                actions_exec[:-1, :] = delta.astype(np.float32)

        # Convert to list-of-arrays as expected by enerdynamics.core.types.Trajectory
        states_list = [states[t].copy() for t in range(states.shape[0])]
        actions_list = [actions_exec[t].copy() for t in range(actions_exec.shape[0])]

        # For debugging: keep both the raw diffusion action and the executed action.
        action0_raw = actions_raw[0].copy()
        action0 = actions_exec[0].copy()
        
        info: Dict[str, Any] = {
            "action0": np.asarray(action0, dtype=np.float32).tolist(),
            "action0_raw": np.asarray(action0_raw, dtype=np.float32).tolist(),
            "action0_source": "derived_from_states" if self.derive_action_from_states else "diffusion_action_slice",
            "goal_xy": np.asarray(self.goal_xy, dtype=np.float32).tolist(),
            "pos_idx": [int(self.pos_idx[0]), int(self.pos_idx[1])],
            "safety": "cbf_qp_invariance" if self._projector is not None else "none",
        }
        if diffusion_paths is not None:
            info["diffusion_paths_shape"] = list(np.asarray(diffusion_paths).shape)

        return {
            "states": states_list,
            "actions": actions_list,
            "info": info,
        }

