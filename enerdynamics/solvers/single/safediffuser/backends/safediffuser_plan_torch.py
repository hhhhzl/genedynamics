from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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

        enable_safety = bool(self.plan_config.get("enable_safety", True))
        if enable_safety:
            cfg_path = str(self.plan_config.get("safediffuser_config_path", "enerdynamics/solvers/single/safediffuser/config/avoiding_d3il.yaml"))
            correct_all_steps = bool(self.plan_config.get("correct_all_steps", False))
            projector = AvoidingCBFQPCorrector(
                normalizer=normalizer,
                config=AvoidingCBFConfig(
                    constraint_config_path=cfg_path,
                    exp=str(self.plan_config.get("exp", "avoiding-d3il")),
                    correct_all_steps=correct_all_steps,
                ),
            )

        self._projector = projector if projector is not None else None
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
        actions = np.asarray(trajectories.actions[self.which_trajectory], dtype=np.float32)
        states = np.asarray(trajectories.observations[self.which_trajectory], dtype=np.float32)

        # Convert to list-of-arrays as expected by enerdynamics.core.types.Trajectory
        states_list = [states[t].copy() for t in range(states.shape[0])]
        actions_list = [actions[t].copy() for t in range(actions.shape[0])]
        action0 = all_sampled_action[max(0, min(int(self.which_trajectory), batch_size - 1)), 0]
        
        info: Dict[str, Any] = {
            "action0": np.asarray(action0, dtype=np.float32).tolist(),
            "safety": "cbf_qp_invariance" if self._projector is not None else "none",
        }
        if diffusion_paths is not None:
            info["diffusion_paths_shape"] = list(np.asarray(diffusion_paths).shape)

        return {
            "states": states_list,
            "actions": actions_list,
            "info": info,
        }

