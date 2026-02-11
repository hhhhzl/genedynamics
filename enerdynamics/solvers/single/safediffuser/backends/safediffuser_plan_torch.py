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
        constraint_manager: Any = None,
        constraint_pipeline: Any = None,
    ):
        self.env = env
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.plan_config = plan_config
        self.device = device
        self.seed = int(seed)
        self.goal_xy = None if goal_xy is None else np.asarray(goal_xy, dtype=np.float32).reshape(2)
        self.constraint_manager = constraint_manager
        self.constraint_pipeline = constraint_pipeline
        self.native_9d: bool = bool(self.plan_config.get("native_9d", False))

        # SafeDiffuser execution details.
        # For avoiding-d3il, we typically constrain/correct the XY *position* components in obs.
        # Default (0,1) corresponds to obs = [x_des, y_des, x, y] (desired position components).
        self.pos_idx: Tuple[int, int] = tuple(self.plan_config.get('safediffuser_pos_idx', (0, 1)))  # type: ignore
        self.derive_action_from_states: bool = bool(self.plan_config.get('derive_action_from_states', True))
        self.use_framework_constraints: bool = bool(self.plan_config.get("use_framework_constraints", True))

        self._policy = None
        self._projector = None
        self._stepper = None

    def _apply_framework_constraints(
        self,
        states_list: list[np.ndarray],
        actions_list: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], Dict[str, Any]]:
        if not self.use_framework_constraints:
            return states_list, actions_list, {"framework_constraint_impl": "disabled"}

        if self.constraint_pipeline is None and self.constraint_manager is None:
            return states_list, actions_list, {"framework_constraint_impl": "none"}

        if len(states_list) < 2 or len(actions_list) < 1:
            return states_list, actions_list, {"framework_constraint_impl": "skipped_short_traj"}

        try:
            from enerdynamics.core.types import Trajectory

            # Trajectory requires len(states) == len(actions) + 1.
            n_actions_nominal = min(len(actions_list), len(states_list) - 1)
            nominal = Trajectory(
                states=[np.asarray(s, dtype=np.float32) for s in states_list],
                actions=[np.asarray(a, dtype=np.float32) for a in actions_list[:n_actions_nominal]],
                info={},
            )

            repaired = nominal
            out_info: Dict[str, Any] = {}

            if self.constraint_pipeline is not None:
                from enerdynamics.core.constraints.core.types import ScheduleState

                repaired, pipe_info = self.constraint_pipeline.apply(
                    nominal=nominal,
                    ref=nominal,
                    state=ScheduleState(k=0, K=1),
                )
                out_info["framework_constraint_impl"] = "constraint_pipeline"
                out_info["framework_constraint_info_keys"] = (
                    list(pipe_info.keys()) if isinstance(pipe_info, dict) else []
                )
            elif (
                self.constraint_manager is not None
                and hasattr(self.constraint_manager, "has_hard")
                and self.constraint_manager.has_hard()
            ):
                repaired = self.constraint_manager.project_hard(
                    nominal, step=0, total_steps=1
                )
                out_info["framework_constraint_impl"] = "constraint_manager_hard"
            else:
                out_info["framework_constraint_impl"] = "none_active"
                return states_list, actions_list, out_info

            repaired_states = [np.asarray(s, dtype=np.float32) for s in repaired.states]
            repaired_actions = [np.asarray(a, dtype=np.float32) for a in repaired.actions]

            # Keep SafeDiffuser's original action-list length contract.
            merged_actions: list[np.ndarray] = []
            for i in range(len(actions_list)):
                if i < len(repaired_actions):
                    merged_actions.append(repaired_actions[i].copy())
                else:
                    merged_actions.append(np.asarray(actions_list[i], dtype=np.float32).copy())

            return repaired_states, merged_actions, out_info
        except Exception as exc:
            return states_list, actions_list, {
                "framework_constraint_impl": "error",
                "framework_constraint_error": str(exc),
            }

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
                    runtime_obstacles=self.plan_config.get("obstacles"),
                    runtime_obstacle_config=self.plan_config.get("obstacle_config", {}),
                    align_constraints_with_framework=bool(
                        self.plan_config.get("align_constraints_with_framework", True)
                    ),
                    disable_halfspace_when_aligned=bool(
                        self.plan_config.get("disable_halfspace_when_aligned", True)
                    ),
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

    def _build_goal_state_9d(self, x0_9d: np.ndarray) -> np.ndarray:
        """
        Build a 9D goal state aligned with D3ILAvoiding9D state:
          [x, y, q1..q7]
        Keep joints from x0 and only set XY to the task goal.
        """
        x0 = np.asarray(x0_9d, dtype=np.float32).reshape(-1).copy()
        if x0.size < 9:
            raise ValueError(f"9D goal build expects x0 size >= 9, got {x0.shape}")
        goal_xy = self.goal_xy
        if goal_xy is None:
            goal_xy = np.asarray(getattr(self.env, "target"), dtype=np.float32).reshape(-1)[:2]
        x0[0] = float(goal_xy[0])
        x0[1] = float(goal_xy[1])
        return x0[:9].astype(np.float32)

    def _plan_core_4d(self, x0_4d: np.ndarray, *, rng_key: Any | None = None) -> Dict[str, Any]:
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
        
        x0 = np.asarray(x0_4d, dtype=np.float32).reshape(-1).copy()
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
        states_list, actions_list, fw_info = self._apply_framework_constraints(states_list, actions_list)

        # For debugging: keep both the raw diffusion action and the executed action.
        action0_raw = actions_raw[0].copy()
        action0 = np.asarray(actions_list[0], dtype=np.float32).copy()
        goal_xy_info = (
            np.asarray(self.goal_xy, dtype=np.float32).tolist()
            if self.goal_xy is not None
            else np.asarray(getattr(self.env, "target"), dtype=np.float32).reshape(-1)[:2].tolist()
        )
        
        info: Dict[str, Any] = {
            "action0": np.asarray(action0, dtype=np.float32).tolist(),
            "action0_raw": np.asarray(action0_raw, dtype=np.float32).tolist(),
            "action0_source": "derived_from_states" if self.derive_action_from_states else "diffusion_action_slice",
            "goal_xy": goal_xy_info,
            "pos_idx": [int(self.pos_idx[0]), int(self.pos_idx[1])],
            "safety": "cbf_qp_invariance" if self._projector is not None else "none",
        }
        info.update(fw_info)
        if diffusion_paths is not None:
            info["diffusion_paths_shape"] = list(np.asarray(diffusion_paths).shape)

        return {
            "states": states_list,
            "actions": actions_list,
            "info": info,
        }

    def _plan_core_9d(self, x0_9d: np.ndarray, *, rng_key: Any | None = None) -> Dict[str, Any]:
        self._ensure_ready()

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

        x0 = np.asarray(x0_9d, dtype=np.float32).reshape(-1).copy()
        if x0.size < 9:
            raise ValueError(f"Native 9D SafeDiffuser expects x0 size >= 9, got {x0.shape}")
        cond[0] = x0[:9]
        cond[horizon - 1] = self._build_goal_state_9d(x0)

        batch_size = int(self.plan_config.get("batch_size", 4))
        _all_sampled_action, trajectories, diffusion_paths = self._policy(
            cond,
            batch_size=batch_size,
            horizon=horizon,
            return_diffusion=bool(self.plan_config.get("return_diffusion", True)),
        )

        actions_raw = np.asarray(trajectories.actions[self.which_trajectory], dtype=np.float32)
        states = np.asarray(trajectories.observations[self.which_trajectory], dtype=np.float32)
        actions_exec = actions_raw

        states_list = [states[t].copy() for t in range(states.shape[0])]
        actions_list = [actions_exec[t].copy() for t in range(actions_exec.shape[0])]
        states_list, actions_list, fw_info = self._apply_framework_constraints(states_list, actions_list)

        action0_raw = actions_raw[0].copy()
        action0 = np.asarray(actions_list[0], dtype=np.float32).copy()
        goal_xy_info = (
            np.asarray(self.goal_xy, dtype=np.float32).tolist()
            if self.goal_xy is not None
            else np.asarray(getattr(self.env, "target"), dtype=np.float32).reshape(-1)[:2].tolist()
        )
        info: Dict[str, Any] = {
            "action0": np.asarray(action0, dtype=np.float32).tolist(),
            "action0_raw": np.asarray(action0_raw, dtype=np.float32).tolist(),
            "action0_source": "diffusion_action_slice",
            "goal_xy": goal_xy_info,
            "native_9d": True,
            "safety": "cbf_qp_invariance" if self._projector is not None else "none",
        }
        info.update(fw_info)
        if diffusion_paths is not None:
            info["diffusion_paths_shape"] = list(np.asarray(diffusion_paths).shape)

        return {
            "states": states_list,
            "actions": actions_list,
            "info": info,
        }

    def plan(self, x0: np.ndarray, *, rng_key: Any | None = None) -> Dict[str, Any]:
        """
        Plan a full horizon trajectory.

        Accepts:
        - native_9d=False: 4D avoiding state or 9D state compressed to 4D conditions
        - native_9d=True: full 9D avoiding state [x, y, q1..q7]
        """
        x = np.asarray(x0, dtype=np.float32).reshape(-1)
        if self.native_9d:
            if x.size < 9:
                raise ValueError(f"Native 9D SafeDiffuser expects x0 size >= 9, got {x.shape}")
            return self._plan_core_9d(x, rng_key=rng_key)
        if x.size == 4:
            x4 = x
        elif x.size >= 2:
            goal_xy = self.goal_xy
            if goal_xy is None:
                goal_xy = np.asarray(getattr(self.env, "target"), dtype=np.float32).reshape(-1)[:2]
            x4 = np.array([float(goal_xy[0]), float(goal_xy[1]), float(x[0]), float(x[1])], dtype=np.float32)
        else:
            raise ValueError(f"SafeDiffuser backend expects x0 with at least 2 dims, got {x.shape}")
        return self._plan_core_4d(x4, rng_key=rng_key)
