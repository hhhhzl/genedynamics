from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SafeDiffuserPlanResult:
    states: list[np.ndarray]
    actions: list[np.ndarray]
    info: Dict[str, Any]


class SafeDiffuserBackendTorch:
    """
    Torch backend that delegates planning to external SafeDiffuser Policy.

    Notes:
    - Expects a SafeDiffuser `Policy` instance (from external repo) that supports:
        action, samples, diffusion_paths, safe1, safe2, elbo = policy(cond, batch_size=B)
      where `samples.actions` and `samples.observations` are numpy arrays.
    - We treat `samples.observations` as the planned state trajectory.
    - We treat `samples.actions` as the planned action trajectory.
    """

    def __init__(
        self,
        *,
        env: Any,
        checkpoint_dir: str,
        epoch: int | str = "latest",
        device: str = "cuda:0",
        batch_size: int = 8,
        goal_xy: Optional[np.ndarray] = None,
    ):
        self.env = env
        self.checkpoint_dir = checkpoint_dir
        self.epoch = epoch
        self.device = device
        # Horizon is fixed by the diffusion checkpoint architecture.
        self.horizon: int | None = None
        self.batch_size = int(batch_size)
        self.goal_xy = None if goal_xy is None else np.asarray(goal_xy, dtype=np.float32).reshape(2)

        self._policy = None

    def _ensure_loaded(self) -> None:
        if self._policy is not None:
            return
        from enerdynamics.solvers.single.safediffuser.diffuser_utils.serialization import (
            load_planning_checkpoint,
        )
        from enerdynamics.solvers.single.safediffuser.patch.policy import Policy
        from enerdynamics.solvers.single.safediffuser.patch.avoiding_cbf_qp import (
            AvoidingCBFConfig,
            AvoidingCBFQPCorrector,
        )

        loaded = load_planning_checkpoint(
            self.checkpoint_dir,
            epoch=self.epoch,
            device=self.device,
        )
        diffusion = loaded.ema
        normalizer = loaded.normalizer

        # Derive horizon from checkpoint (authoritative).
        ckpt_horizon = int(getattr(diffusion, "horizon", 0))
        if ckpt_horizon <= 0:
            raise ValueError("Loaded diffusion checkpoint has invalid horizon")
        self.horizon = ckpt_horizon

        projector = None
        enable_cbf = bool(getattr(self.env, "enable_safety_cbf", False))
        if enable_cbf:
            cfg_path = getattr(self.env, "constraint_config_path", "") or str(
                Path(__file__).resolve().parents[1]
                / "config"
                / "avoiding_d3il.yaml"
            )
            projector = AvoidingCBFQPCorrector(
                normalizer=normalizer,
                config=AvoidingCBFConfig(
                    constraint_config_path=cfg_path,
                    exp="avoiding-d3il",
                    halfspace_variant=getattr(self.env, "halfspace_variant", None),
                ),
            )

        self._policy = Policy(diffusion, normalizer, projector=projector)

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

        self._ensure_loaded()
        assert self.horizon is not None

        # Best-effort seeding (SafeDiffuser uses torch internally).
        if rng_key is not None:
            try:
                import torch

                seed = int(getattr(rng_key, "integers", lambda low, high: 0)(0, 2**31 - 1))
                torch.manual_seed(seed)
            except Exception:
                pass

        cond: Dict[int, np.ndarray] = {}
        cond[0] = x0
        cond[self.horizon - 1] = self._build_goal_state_4d()

        action0, samples, diffusion_paths, safe1, safe2, elbo = self._policy(
            cond, batch_size=self.batch_size
        )

        # SafeDiffuser returns: samples.actions [B,H,A], samples.observations [B,H,O]
        actions = np.asarray(samples.actions[0], dtype=np.float32)
        states = np.asarray(samples.observations[0], dtype=np.float32)

        # Convert to list-of-arrays as expected by enerdynamics.core.types.Trajectory
        states_list = [states[t].copy() for t in range(states.shape[0])]
        actions_list = [actions[t].copy() for t in range(actions.shape[0])]

        info: Dict[str, Any] = {
            "safe1": float(safe1) if np.isscalar(safe1) else safe1,
            "safe2": float(safe2) if np.isscalar(safe2) else safe2,
            "elbo": float(elbo) if np.isscalar(elbo) else elbo,
            "action0": np.asarray(action0, dtype=np.float32).tolist(),
        }
        # diffusion_paths can be large; keep only a minimal preview
        try:
            dp = np.asarray(diffusion_paths)
            info["diffusion_paths_shape"] = list(dp.shape)
        except Exception:
            pass

        return {
            "states": states_list,
            "actions": actions_list,
            "info": info,
        }

