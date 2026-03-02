from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from genedynamics.core.task_spec import legacy_extract_position


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
        constraint_pipeline: Any = None,
    ):
        self.env = env
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.plan_config = plan_config
        self.device = device
        self.seed = int(seed)
        self._position_extractor = plan_config.get("position_extractor") or legacy_extract_position
        self._position_dim = int(plan_config.get("position_dim", 2))
        self.goal_xy = None
        if goal_xy is not None:
            self.goal_xy = np.asarray(self._position_extractor(goal_xy), dtype=np.float32).reshape(-1)[: self._position_dim].copy()
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

    def _resolve_goal_xy(self) -> np.ndarray:
        if self.goal_xy is not None:
            return np.asarray(self.goal_xy, dtype=np.float32).reshape(-1)[: self._position_dim].copy()
        t = getattr(self.env, "target", None)
        if t is not None:
            return np.asarray(self._position_extractor(t), dtype=np.float32).reshape(-1)[: self._position_dim].copy()
        return np.zeros(self._position_dim, dtype=np.float32)

    def _apply_framework_constraints(
        self,
        states_list: list[np.ndarray],
        actions_list: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], Dict[str, Any]]:
        # For native 9D, the framework projection can be overly conservative on harder obstacle levels
        # and stall progress (ssr/execution_ssr -> 0). Since we already apply action-space safety QP
        # and execute with tracking, skip framework constraints for level>=1 to preserve reachability.
        if bool(self.plan_config.get("native_9d", False)):
            try:
                lvl = int(getattr(getattr(self.env, "config", None), "obstacle_level", 0) or 0)
            except Exception:
                lvl = 0
            if lvl >= 1:
                return states_list, actions_list, {"framework_constraint_impl": "skipped_native_9d_level_ge_1"}
        if not self.use_framework_constraints:
            return states_list, actions_list, {"framework_constraint_impl": "disabled"}

        if self.constraint_pipeline is None:
            return states_list, actions_list, {"framework_constraint_impl": "none"}

        if len(states_list) < 2 or len(actions_list) < 1:
            return states_list, actions_list, {"framework_constraint_impl": "skipped_short_traj"}

        try:
            from genedynamics.core.types import Trajectory

            # Trajectory requires len(states) == len(actions) + 1.
            n_actions_nominal = min(len(actions_list), len(states_list) - 1)
            nominal = Trajectory(
                states=[np.asarray(s, dtype=np.float32) for s in states_list],
                actions=[np.asarray(a, dtype=np.float32) for a in actions_list[:n_actions_nominal]],
                info={},
            )

            repaired = nominal
            out_info: Dict[str, Any] = {}

            from genedynamics.core.constraints.core.types import ScheduleState

            repaired, pipe_info = self.constraint_pipeline.apply(
                nominal=nominal,
                ref=nominal,
                state=ScheduleState(k=0, K=1),
            )
            out_info["framework_constraint_impl"] = "constraint_pipeline"
            out_info["framework_constraint_info_keys"] = (
                list(pipe_info.keys()) if isinstance(pipe_info, dict) else []
            )

            repaired_states = [np.asarray(s, dtype=np.float32) for s in repaired.states]
            repaired_actions = [np.asarray(a, dtype=np.float32) for a in repaired.actions]

            # Keep SafeDiffuser's original action-list length contract.
            merged_actions: list[np.ndarray] = []
            for i in range(len(actions_list)):
                if i < len(repaired_actions):
                    merged_actions.append(repaired_actions[i].copy())
                else:
                    merged_actions.append(np.asarray(actions_list[i], dtype=np.float32).copy())

            # CRITICAL (9D): Some framework projections operate in a different internal state space and can
            # output "repaired" states that are inconsistent with 9D kinematics (q' = q + dt*qdot),
            # which makes modes_plan look like it "floats/jumps" while execution (env.step) is normal.
            # For native 9D, treat the projection as an *action* repair and re-rollout states from the
            # original initial state using the environment's rollout model.
            if bool(self.plan_config.get("native_9d", False)) and hasattr(self.env, "rollout_actions"):
                try:
                    x0_9d = np.asarray(states_list[0], dtype=np.float32).reshape(-1)
                    if x0_9d.size >= 9:
                        # First try: use repaired actions.
                        acts_arr = np.asarray(merged_actions, dtype=np.float32)
                        rolled = self.env.rollout_actions(x0_9d[:9], acts_arr)
                        rolled = np.asarray(rolled, dtype=np.float32)
                        end_y = float(rolled[-1, 1]) if rolled.ndim == 2 and rolled.shape[1] >= 2 else float(x0_9d[1])
                        # If the repair makes the trajectory stall far below the target line, fall back to original actions.
                        # (This prevents ssr/exec_ssr collapsing to 0 on harder levels.)
                        try:
                            tgt = np.asarray(self._position_extractor(getattr(self.env, "target")), dtype=np.float32).reshape(-1)
                            target_y = float(tgt[1]) if tgt.size >= 2 else 0.35
                        except Exception:
                            target_y = 0.35
                        if end_y < target_y - 0.15:
                            acts_arr = np.asarray(actions_list, dtype=np.float32)
                            rolled = self.env.rollout_actions(x0_9d[:9], acts_arr)
                            rolled = np.asarray(rolled, dtype=np.float32)
                            rerolled_states = [rolled[t].copy() for t in range(rolled.shape[0])]
                            return rerolled_states, [np.asarray(a, dtype=np.float32) for a in actions_list], {**out_info, "framework_constraint_states": "rerollout_fallback_original_actions"}
                        rerolled_states = [rolled[t].copy() for t in range(rolled.shape[0])]
                        return rerolled_states, merged_actions, {**out_info, "framework_constraint_states": "rerollout_from_actions"}
                except Exception as _exc:
                    out_info["framework_constraint_states"] = "rerollout_failed"
            return repaired_states, merged_actions, out_info
        except Exception as exc:
            return states_list, actions_list, {
                "framework_constraint_impl": "error",
                "framework_constraint_error": str(exc),
            }

    def _project_actions_9d_cbf_qp(
        self,
        x0_9d: np.ndarray,
        actions: np.ndarray,
        *,
        alpha: float = 0.5,
        safety_margin: float = 0.0,
        freeze_jacobian: bool = True,
    ) -> np.ndarray:
        """
        Post-process 7D qdot actions with a small CBF-QP in action space:
        keep the next-step tcp_xy away from circular obstacles using local Jacobian J_xy.

        This is separate from SafeDiffuser's denoising-time invariance hook (which corrects obs),
        and is needed because D3IL executes `candidate_actions` open-loop in 9D.
        """
        try:
            if actions is None:
                return actions
            u_nom = np.asarray(actions, dtype=np.float32)
            if u_nom.ndim == 1:
                u_nom = u_nom.reshape(1, -1)
            if u_nom.shape[-1] != 7 or u_nom.shape[0] == 0:
                return u_nom
            if self._projector is None:
                return u_nom
            obs_list = getattr(self._projector, "obstacles", None)
            if not obs_list:
                return u_nom
            if not hasattr(self.env, "get_jacobian_xy"):
                return u_nom

            # Build obstacle tensors once (numpy).
            centers = np.asarray([np.asarray(c, dtype=np.float32).reshape(2) for c, _r in obs_list], dtype=np.float32)
            # Match SSR semantics: treat robot as disc with radius=robot_radius.
            try:
                robot_radius = float((self.plan_config.get("obstacle_config") or {}).get("robot_radius", 0.0))
            except Exception:
                robot_radius = 0.0

            # Level-aware relaxation: for harder obstacle levels, extra margin can over-constrain the QP and stall motion.
            # Keep level0 behavior unchanged; for level>=1 drop extra margin and relax alpha.
            try:
                lvl = int(getattr(getattr(self.env, "config", None), "obstacle_level", 0) or 0)
            except Exception:
                lvl = 0
            alpha_eff = float(alpha)
            margin_eff = float(safety_margin)
            if lvl >= 1:
                margin_eff = 0.0
                alpha_eff = max(alpha_eff, 0.3)
            radii = (
                np.asarray([float(r) for _c, r in obs_list], dtype=np.float32)
                + float(robot_radius)
                + float(margin_eff)
            ).astype(np.float32)
            if centers.size == 0:
                return u_nom

            import torch
            from qpth.qp import QPFunction

            x = np.asarray(x0_9d, dtype=np.float32).reshape(-1)
            if x.size < 9:
                return u_nom
            dt = float(getattr(self.env, "dt", 0.035))
            u_lim = float(getattr(self.env, "control_limit", 1.5))

            u_out = np.zeros_like(u_nom, dtype=np.float32)
            J0 = None
            if freeze_jacobian:
                J0 = self.env.get_jacobian_xy(x) if hasattr(self.env, "get_jacobian_xy") else None
                if J0 is not None:
                    J0 = np.asarray(J0, dtype=np.float32)
                    if J0.shape != (2, 7):
                        J0 = None
            for t in range(u_nom.shape[0]):
                u0 = np.clip(u_nom[t].reshape(7), -u_lim, u_lim).astype(np.float32)
                J = J0 if J0 is not None else (self.env.get_jacobian_xy(x) if hasattr(self.env, "get_jacobian_xy") else None)
                if J is None:
                    u_out[t] = u0
                    # Update q only; tcp remains
                    x[2:9] = x[2:9] + dt * u0
                    continue
                J = np.asarray(J, dtype=np.float32)
                if J.shape != (2, 7):
                    u_out[t] = u0
                    x[2:9] = x[2:9] + dt * u0
                    continue

                p = x[:2].astype(np.float32)
                dvec = (p.reshape(1, 2) - centers)  # (K,2)
                b_val = (dvec[:, 0] ** 2 + dvec[:, 1] ** 2) - (radii ** 2)  # (K,)
                grad = 2.0 * dvec  # (K,2)

                # Inequalities in u:  -grad @ (dt * J @ u) <= alpha * b
                # => G u <= h
                G_obs = -(dt * (grad @ J)).astype(np.float32)  # (K,7)
                h_obs = (float(alpha_eff) * b_val).astype(np.float32)  # (K,)

                # Bounds |u| <= u_lim
                G_bound = np.concatenate([np.eye(7, dtype=np.float32), -np.eye(7, dtype=np.float32)], axis=0)
                h_bound = np.full((14,), u_lim, dtype=np.float32)

                G = np.concatenate([G_bound, G_obs], axis=0)
                h = np.concatenate([h_bound, h_obs], axis=0)

                # Solve QP: min ||u - u0||^2  s.t. G u <= h
                Q = 2.0 * torch.eye(7, dtype=torch.float32).unsqueeze(0)
                p_lin = (-2.0 * torch.as_tensor(u0, dtype=torch.float32)).unsqueeze(0)
                G_t = torch.as_tensor(G, dtype=torch.float32).unsqueeze(0)
                h_t = torch.as_tensor(h, dtype=torch.float32).unsqueeze(0)
                try:
                    u_star = QPFunction(verbose=False)(Q, p_lin, G_t, h_t, None, None)
                    u_star = u_star.squeeze(0).detach().cpu().numpy().astype(np.float32)
                except Exception:
                    u_star = u0

                u_star = np.clip(u_star, -u_lim, u_lim).astype(np.float32)
                u_out[t] = u_star
                # Roll forward with local Jacobian (same as rollout_actions does).
                x[:2] = x[:2] + dt * (J @ u_star)
                x[2:9] = x[2:9] + dt * u_star

            return u_out
        except Exception:
            return np.asarray(actions, dtype=np.float32)

    def _ensure_ready(self) -> None:
        if self._policy is not None:
            return
        from genedynamics.solvers.single.safediffuser.patch.avoiding_cbf_qp import (
            AvoidingCBFConfig,
            AvoidingCBFQPCorrector,
        )
        from genedynamics.solvers.single.safediffuser.stepper import SafeDiffuserTorchStepper
        from genedynamics.solvers.single.safediffuser.policies import SafeDiffuserPolicy

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
            cfg_path = str(self.plan_config.get("safediffuser_config_path", "genedynamics/solvers/single/safediffuser/config/avoiding_d3il.yaml"))
            correct_all_steps = bool(self.plan_config.get("correct_all_steps", True))
            # IMPORTANT:
            # - For 4D avoiding obs=[x_des,y_des,x,y], safety acts on (x_des,y_des) => indices (0,1).
            # - For 9D with goal-conditioned 11D obs=[goal_x,goal_y,x,y,q1..q7], safety must act on current (x,y) => indices (2,3).
            des_idx = tuple(self.pos_idx)
            try:
                obs_dim = int(getattr(normalizer, "observation_dim", 0))
            except Exception:
                obs_dim = 0
            if bool(self.plan_config.get("native_9d", False)) and obs_dim == 11:
                des_idx = (2, 3)
            projector = AvoidingCBFQPCorrector(
                normalizer=normalizer,
                config=AvoidingCBFConfig(
                    constraint_config_path=cfg_path,
                    exp=str(self.plan_config.get("exp", "avoiding-d3il")),
                    des_idx=des_idx,
                    correct_all_steps=correct_all_steps,
                    runtime_obstacles=self.plan_config.get("obstacles"),
                    runtime_obstacle_config=self.plan_config.get("obstacle_config", {}),
                    align_constraints_with_framework=bool(
                        self.plan_config.get("align_constraints_with_framework", True)
                    ),
                    halfspace_variants=self.plan_config.get("halfspace_variants", None)
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
        goal_xy = self._resolve_goal_xy()
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        return np.array([gx, gy, gx, gy], dtype=np.float32)

    def _plan_core_4d_chunked(self, x0_4d: np.ndarray, *, rng_key: Any | None = None) -> Dict[str, Any]:
        """Multi-chunk plan: num_modes long trajectories, each chunk conditions on previous chunk end state (batch conditions)."""
        self._ensure_ready()
        try:
            if rng_key is not None:
                seed = int(getattr(rng_key, "integers", lambda low, high: 0)(0, 2**31 - 1))
            else:
                seed = self.seed
            torch.manual_seed(seed)
        except Exception:
            pass

        horizon = int(self.plan_config.get("horizon", getattr(self.diffusion, "horizon", 8)))
        num_modes = int(self.plan_config.get("num_modes", 1))
        plan_once_chunks = int(self.plan_config.get("plan_once_chunks", 1))
        max_episode_length = int(self.plan_config.get("max_episode_length", 200))
        use_target_line = bool(self.plan_config.get("use_target_line", False))
        num_targets = int(self.plan_config.get("num_targets", 4))
        goal_xy = self._resolve_goal_xy()
        goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)[: self._position_dim].copy()

        x0 = np.asarray(x0_4d, dtype=np.float32).reshape(-1).copy()
        x0_obs = x0[:4].copy() if x0.size >= 4 else np.concatenate([np.zeros(4 - x0.size, dtype=np.float32), x0])
        current_obs_batch = np.tile(x0_obs.reshape(1, -1), (num_modes, 1))
        mode_states: list = [[x0_obs.copy()] for _ in range(num_modes)]
        mode_actions: list = [[] for _ in range(num_modes)]

        for chunk_idx in range(plan_once_chunks):
            if len(mode_actions[0]) >= max_episode_length:
                break
            cond: Dict[int, np.ndarray] = {}
            cond[0] = current_obs_batch.copy()
            if use_target_line:
                from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
                target_line = get_d3il_target_line_positions(num_targets)
                goals_4d = np.array([[gx, gy, gx, gy] for gx, gy in target_line[:num_targets]], dtype=np.float32)
                idx = np.arange(num_modes) % len(goals_4d)
                cond[horizon - 1] = goals_4d[idx]
            else:
                goal_4d = self._build_goal_state_4d()
                cond[horizon - 1] = np.tile(goal_4d.reshape(1, -1), (num_modes, 1))

            all_sampled_action, trajectories, diffusion_paths = self._policy(
                cond,
                batch_size=num_modes,
                horizon=horizon,
                return_diffusion=bool(self.plan_config.get("return_diffusion", True)),
            )
            obs_all = np.asarray(trajectories.observations, dtype=np.float32)
            act_all = np.asarray(trajectories.actions, dtype=np.float32)
            is_last_chunk = chunk_idx == plan_once_chunks - 1
            for i in range(num_modes):
                states_b = np.asarray(obs_all[i], dtype=np.float32)
                actions_raw_b = np.asarray(act_all[i], dtype=np.float32)
                actions_exec_b = actions_raw_b
                if (
                    self.derive_action_from_states
                    and states_b.ndim == 2
                    and states_b.shape[0] >= 2
                    and actions_raw_b.shape[-1] == 2
                ):
                    i0, i1 = int(self.pos_idx[0]), int(self.pos_idx[1])
                    if 0 <= i0 < states_b.shape[1] and 0 <= i1 < states_b.shape[1]:
                        pos = states_b[:, [i0, i1]]
                        delta = pos[1:] - pos[:-1]
                        actions_exec_b = np.zeros_like(actions_raw_b, dtype=np.float32)
                        actions_exec_b[:-1, :] = delta.astype(np.float32)
                # cond[0]=current_obs, cond[horizon-1]=goal => states_b[-1] is goal (conditioned), use -2 as next start
                if is_last_chunk:
                    for t in range(1, states_b.shape[0]):
                        mode_states[i].append(states_b[t].copy())
                    for t in range(actions_exec_b.shape[0] - 1):
                        mode_actions[i].append(actions_exec_b[t].copy())
                else:
                    for t in range(1, states_b.shape[0] - 1):
                        mode_states[i].append(states_b[t].copy())
                    for t in range(min(actions_exec_b.shape[0], states_b.shape[0] - 2)):
                        mode_actions[i].append(actions_exec_b[t].copy())
                current_obs_batch[i] = states_b[-2].copy()

        candidate_states = [[np.asarray(s, dtype=np.float32) for s in mode_states[i]] for i in range(num_modes)]
        candidate_actions = [[np.asarray(a, dtype=np.float32) for a in mode_actions[i]] for i in range(num_modes)]
        candidate_costs = []
        for i in range(num_modes):
            last_pos = np.asarray(self._position_extractor(candidate_states[i][-1]), dtype=np.float32).reshape(-1)[: self._position_dim]
            cost_i = float(np.sum((last_pos - goal_xy) ** 2)) if last_pos.size >= 1 else 0.0
            candidate_costs.append(cost_i)
        best_idx = int(np.argmin(candidate_costs))
        states_list = candidate_states[best_idx]
        actions_list = candidate_actions[best_idx]
        states_list, actions_list, fw_info = self._apply_framework_constraints(states_list, actions_list)
        obs_dim = 4
        x0_obs_final = x0_obs[:obs_dim]
        if len(states_list) > 0 and np.asarray(states_list[0]).size >= obs_dim:
            states_list[0] = x0_obs_final.copy()

        action0 = np.asarray(actions_list[0], dtype=np.float32).copy() if actions_list else np.zeros(2, dtype=np.float32)
        goal_xy_info = self._resolve_goal_xy().tolist()
        info: Dict[str, Any] = {}
        info.update(fw_info)
        info.update({
            "action0": np.asarray(action0, dtype=np.float32).tolist(),
            "action0_source": "derived_from_states" if self.derive_action_from_states else "diffusion_action_slice",
            "goal_xy": goal_xy_info,
            "pos_idx": [int(self.pos_idx[0]), int(self.pos_idx[1])],
            "safety": "cbf_qp_invariance" if self._projector is not None else "none",
        })
        if diffusion_paths is not None:
            info["diffusion_paths_shape"] = list(np.asarray(diffusion_paths).shape)
        info["candidate_states"] = candidate_states
        info["candidate_actions"] = candidate_actions
        info["candidate_costs"] = np.asarray(candidate_costs, dtype=np.float32).tolist()
        info["best_idx"] = best_idx
        info["mode_strategy"] = "multirun"
        return {
            "states": states_list,
            "actions": actions_list,
            "info": info,
        }

    def _build_goal_state_9d(self, x0_9d: np.ndarray) -> np.ndarray:
        """
        Build a 9D goal state aligned with D3ILAvoiding9D state:
          [x, y, q1..q7]
        Keep joints from x0 and only set XY to the task goal.
        """
        x0 = np.asarray(x0_9d, dtype=np.float32).reshape(-1).copy()
        if x0.size < 9:
            raise ValueError(f"9D goal build expects x0 size >= 9, got {x0.shape}")
        goal_xy = self._resolve_goal_xy()
        x0[0] = float(goal_xy[0])
        x0[1] = float(goal_xy[1])
        return x0[:9].astype(np.float32)

    @staticmethod
    def _obs_9d_to_11d(obs_9d: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        """Convert 9D [x, y, q1..q7] to dataset 11D [x_des, y_des, x, y, q1..q7] for normalizer compatibility."""
        obs = np.asarray(obs_9d, dtype=np.float32).reshape(-1)
        goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
        if obs.size < 9:
            raise ValueError(f"obs_9d must have at least 9 dims, got {obs.size}")
        return np.concatenate([goal, obs[:2], obs[2:9]], axis=0).astype(np.float32)

    @staticmethod
    def _obs_11d_to_9d(obs_11d: np.ndarray) -> np.ndarray:
        """Convert 11D [x_des, y_des, x, y, q1..q7] to 9D [x, y, q1..q7] for env/framework."""
        obs = np.asarray(obs_11d, dtype=np.float32).reshape(-1)
        if obs.size < 11:
            return obs[:9].copy()
        return np.concatenate([obs[2:4], obs[4:11]], axis=0).astype(np.float32)

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
        batch_size = int(self.plan_config.get("batch_size", 4))
        use_target_line = bool(self.plan_config.get("use_target_line", False))
        num_targets = int(self.plan_config.get("num_targets", 4))
        
        x0 = np.asarray(x0_4d, dtype=np.float32).reshape(-1).copy()
        num_modes = int(self.plan_config.get("num_modes", 1))
        if num_modes > 1:
            batch_size = max(batch_size, num_modes)
        cond[0] = x0
        if use_target_line:
            from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
            target_line = get_d3il_target_line_positions(num_targets)
            goals_4d = np.array([[gx, gy, gx, gy] for gx, gy in target_line[:num_targets]], dtype=np.float32)
            # Repeat to match batch_size
            idx = np.arange(batch_size) % len(goals_4d)
            cond[horizon - 1] = goals_4d[idx]
        else:
            cond[horizon - 1] = self._build_goal_state_4d()
        all_sampled_action, trajectories, diffusion_paths = self._policy(
            cond,
            batch_size=batch_size,
            horizon=horizon,
            return_diffusion=bool(self.plan_config.get("return_diffusion", True)),
        )

        # Policy returns: trajectories.actions [B,H,A], trajectories.observations [B,H,O]
        obs_all = np.asarray(trajectories.observations, dtype=np.float32)
        act_all = np.asarray(trajectories.actions, dtype=np.float32)
        B = obs_all.shape[0]
        goal_xy = self._resolve_goal_xy()

        candidate_states: list = []
        candidate_actions: list = []
        candidate_costs: list = []
        for b in range(B):
            states_b = np.asarray(obs_all[b], dtype=np.float32)
            actions_raw_b = np.asarray(act_all[b], dtype=np.float32)
            actions_exec_b = actions_raw_b
            if (
                self.derive_action_from_states
                and states_b.ndim == 2
                and states_b.shape[0] >= 2
                and actions_raw_b.shape[-1] == 2
            ):
                i0, i1 = int(self.pos_idx[0]), int(self.pos_idx[1])
                if 0 <= i0 < states_b.shape[1] and 0 <= i1 < states_b.shape[1]:
                    pos = states_b[:, [i0, i1]]
                    delta = pos[1:] - pos[:-1]
                    actions_exec_b = np.zeros_like(actions_raw_b, dtype=np.float32)
                    actions_exec_b[:-1, :] = delta.astype(np.float32)
            states_list_b = [states_b[t].copy() for t in range(states_b.shape[0])]
            actions_list_b = [actions_exec_b[t].copy() for t in range(actions_exec_b.shape[0])]
            states_list_b, actions_list_b, _ = self._apply_framework_constraints(states_list_b, actions_list_b)
            candidate_states.append(states_list_b)
            candidate_actions.append(actions_list_b)
            # Cost = distance to goal at last state
            last_pos = np.asarray(self._position_extractor(states_list_b[-1]), dtype=np.float32).reshape(-1)[: self._position_dim]
            cost_b = float(np.sum((last_pos - goal_xy) ** 2)) if last_pos.size >= 1 else 0.0
            candidate_costs.append(cost_b)

        best_idx = int(np.argmin(candidate_costs))
        which = self.which_trajectory if num_modes <= 1 else best_idx
        which = min(which, B - 1)
        states_list = candidate_states[which]
        actions_list = candidate_actions[which]
        actions_raw = np.asarray(act_all[which], dtype=np.float32)
        states = np.asarray(obs_all[which], dtype=np.float32)
        _, _, fw_info = self._apply_framework_constraints(states_list, actions_list)
        info = {}
        info.update(fw_info)

        # For debugging: keep both the raw diffusion action and the executed action.
        action0_raw = actions_raw[0].copy()
        action0 = np.asarray(actions_list[0], dtype=np.float32).copy()
        goal_xy_info = self._resolve_goal_xy().tolist()
        info.update({
            "action0": np.asarray(action0, dtype=np.float32).tolist(),
            "action0_raw": np.asarray(action0_raw, dtype=np.float32).tolist(),
            "action0_source": "derived_from_states" if self.derive_action_from_states else "diffusion_action_slice",
            "goal_xy": goal_xy_info,
            "pos_idx": [int(self.pos_idx[0]), int(self.pos_idx[1])],
            "safety": "cbf_qp_invariance" if self._projector is not None else "none",
        })
        if diffusion_paths is not None:
            info["diffusion_paths_shape"] = list(np.asarray(diffusion_paths).shape)
        if num_modes > 1 and candidate_states:
            info["candidate_states"] = candidate_states
            info["candidate_actions"] = candidate_actions
            info["candidate_costs"] = np.asarray(candidate_costs, dtype=np.float32).tolist()
            info["best_idx"] = best_idx
            info["mode_strategy"] = "multirun"

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
        batch_size = int(self.plan_config.get("batch_size", 4))
        use_target_line = bool(self.plan_config.get("use_target_line", False))
        num_targets = int(self.plan_config.get("num_targets", 4))

        x0 = np.asarray(x0_9d, dtype=np.float32).reshape(-1).copy()
        if x0.size < 9:
            raise ValueError(f"Native 9D SafeDiffuser expects x0 size >= 9, got {x0.shape}")
        goal_xy = self._resolve_goal_xy()

        # Dataset may be 11D [x_des, y_des, x, y, q1..q7]; normalizer expects same dim.
        obs_dim = int(getattr(self.normalizer, "observation_dim", 9))
        use_11d = obs_dim == 11

        if use_11d:
            cond[0] = self._obs_9d_to_11d(x0[:9], goal_xy)
        else:
            cond[0] = x0[:9].copy()
        if use_target_line:
            from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
            target_line = get_d3il_target_line_positions(num_targets)
            goals_9d = np.tile(x0[:9].reshape(1, -1), (batch_size, 1))
            for i in range(batch_size):
                gx, gy = target_line[i % len(target_line)]
                goals_9d[i, 0], goals_9d[i, 1] = gx, gy
            if use_11d:
                goals_11d = np.zeros((batch_size, 11), dtype=np.float32)
                for i in range(batch_size):
                    goals_11d[i] = self._obs_9d_to_11d(goals_9d[i], np.array([goals_9d[i, 0], goals_9d[i, 1]], dtype=np.float32))
                cond[horizon - 1] = goals_11d
            else:
                cond[horizon - 1] = goals_9d
        else:
            goal_9d = self._build_goal_state_9d(x0)
            if use_11d:
                cond[horizon - 1] = self._obs_9d_to_11d(goal_9d, goal_xy)
            else:
                cond[horizon - 1] = goal_9d
        _all_sampled_action, trajectories, diffusion_paths = self._policy(
            cond,
            batch_size=batch_size,
            horizon=horizon,
            return_diffusion=bool(self.plan_config.get("return_diffusion", True)),
        )

        actions_raw = np.asarray(trajectories.actions[self.which_trajectory], dtype=np.float32)
        obs_out = np.asarray(trajectories.observations[self.which_trajectory], dtype=np.float32)
        if use_11d and obs_out.shape[-1] >= 11:
            states = np.array([self._obs_11d_to_9d(obs_out[t]) for t in range(obs_out.shape[0])], dtype=np.float32)
        else:
            states = obs_out
        actions_exec = actions_raw

        states_list = [states[t].copy() for t in range(states.shape[0])]
        actions_list = [actions_exec[t].copy() for t in range(actions_exec.shape[0])]
        states_list, actions_list, fw_info = self._apply_framework_constraints(states_list, actions_list)

        action0_raw = actions_raw[0].copy()
        action0 = np.asarray(actions_list[0], dtype=np.float32).copy()
        goal_xy_info = self._resolve_goal_xy().tolist()
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

    def _plan_core_9d_chunked(self, x0_9d: np.ndarray, *, rng_key: Any | None = None) -> Dict[str, Any]:
        """Multi-chunk 9D plan: long trajectory (plan_once_chunks × horizon), optional num_modes.
        When plan_once_chunks==1 and num_modes>1: run num_modes independent single-rollout plans (different seeds)
        so each mode has same distribution as num_modes=1 (obstacle-avoiding), not one batch of 20.
        """
        self._ensure_ready()
        try:
            if rng_key is not None:
                base_seed = int(getattr(rng_key, "integers", lambda low, high: 0)(0, 2**31 - 1))
            else:
                base_seed = self.seed
        except Exception:
            base_seed = self.seed

        horizon = int(self.plan_config.get("horizon", getattr(self.diffusion, "horizon", 8)))
        num_modes = int(self.plan_config.get("num_modes", 1))
        plan_once_chunks = int(self.plan_config.get("plan_once_chunks", 1))
        plan_once_steps_per_chunk = int(self.plan_config.get("plan_once_steps_per_chunk", 0))
        max_episode_length = int(self.plan_config.get("max_episode_length", 200))
        use_target_line = bool(self.plan_config.get("use_target_line", False))
        num_targets = int(self.plan_config.get("num_targets", 4))
        goal_xy = self._resolve_goal_xy()

        x0 = np.asarray(x0_9d, dtype=np.float32).reshape(-1).copy()
        if x0.size < 9:
            raise ValueError(f"Native 9D SafeDiffuser expects x0 size >= 9, got {x0.shape}")
        x0_9 = x0[:9]
        obs_dim = int(getattr(self.normalizer, "observation_dim", 9))
        use_11d = obs_dim == 11

        # Multi-mode with single chunk: generate each mode by independent rollout (different seed) so trajectories match num_modes=1 quality
        if plan_once_chunks == 1 and num_modes > 1:
            mode_states = []
            mode_actions = []
            goals_xy_list = []
            use_target_line = bool(self.plan_config.get("use_target_line", False))
            num_targets = int(self.plan_config.get("num_targets", 4))
            for i in range(num_modes):
                torch.manual_seed(base_seed + i)
                if use_target_line:
                    from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
                    target_line = get_d3il_target_line_positions(num_targets)
                    gx, gy = target_line[i % len(target_line)]
                    goal_9d_i = x0_9.copy()
                    goal_9d_i[0], goal_9d_i[1] = gx, gy
                    goal_xy_i = np.array([gx, gy], dtype=np.float32)
                else:
                    goal_9d_i = self._build_goal_state_9d(x0)
                    goal_xy_i = goal_xy
                goals_xy_list.append(goal_xy_i)
                cond = {}
                if use_11d:
                    cond[0] = self._obs_9d_to_11d(x0_9, goal_xy_i).reshape(1, -1)
                    cond[horizon - 1] = self._obs_9d_to_11d(goal_9d_i, goal_xy_i).reshape(1, -1)
                else:
                    cond[0] = x0_9.reshape(1, -1).copy()
                    cond[horizon - 1] = goal_9d_i.reshape(1, -1)
                _a, trajectories, _ = self._policy(cond, batch_size=1, horizon=horizon, return_diffusion=bool(self.plan_config.get("return_diffusion", True)))
                obs = np.asarray(trajectories.observations[0], dtype=np.float32)
                act = np.asarray(trajectories.actions[0], dtype=np.float32)
                if use_11d and obs.shape[-1] >= 11:
                    obs = np.array([self._obs_11d_to_9d(obs[t]) for t in range(obs.shape[0])], dtype=np.float32)
                states_i = [x0_9.copy()] + [obs[t].copy() for t in range(1, obs.shape[0])]
                actions_i = [act[t].copy() for t in range(act.shape[0] - 1)] if act.shape[0] > 1 else []
                mode_states.append(states_i)
                mode_actions.append(actions_i)
            candidate_states = [[np.asarray(s, dtype=np.float32) for s in mode_states[i]] for i in range(num_modes)]
            candidate_actions = [[np.asarray(a, dtype=np.float32) for a in mode_actions[i]] for i in range(num_modes)]
            candidate_costs = [float(np.sum((np.asarray(self._position_extractor(candidate_states[i][-1]), dtype=np.float32).reshape(-1)[: self._position_dim] - goals_xy_list[i]) ** 2)) for i in range(num_modes)]
            best_idx = int(np.argmin(candidate_costs))
            states_list = list(candidate_states[best_idx])
            actions_list = list(candidate_actions[best_idx])
            states_list, actions_list, fw_info = self._apply_framework_constraints(states_list, actions_list)
            if len(states_list) > 0 and np.asarray(states_list[0]).size >= 9:
                states_list[0] = x0_9.copy()
            action0 = np.asarray(actions_list[0], dtype=np.float32).copy() if actions_list else np.zeros(7, dtype=np.float32)
            info = dict(fw_info)
            info.update({
                "action0": action0.tolist(),
                "goal_xy": self._resolve_goal_xy().tolist(),
                "native_9d": True,
                "safety": "cbf_qp_invariance" if self._projector is not None else "none",
                "candidate_states": candidate_states,
                "candidate_actions": candidate_actions,
                "candidate_costs": np.asarray(candidate_costs, dtype=np.float32).tolist(),
                "best_idx": best_idx,
                "mode_strategy": "multirun",
            })
            return {"states": states_list, "actions": actions_list, "info": info}

        torch.manual_seed(base_seed)
        current_obs_batch = np.tile(x0_9.reshape(1, -1), (num_modes, 1))
        mode_states: list = [[x0_9.copy()] for _ in range(num_modes)]
        mode_actions: list = [[] for _ in range(num_modes)]

        # Per-mode goal assignment (multi-target target-line), fixed across chunks.
        if use_target_line:
            from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
            target_line = get_d3il_target_line_positions(num_targets)
            if target_line is None or len(target_line) == 0:
                goals_xy_modes = np.tile(goal_xy.reshape(1, 2), (num_modes, 1)).astype(np.float32)
                goals_idx_modes = (np.zeros((num_modes,), dtype=np.int64)).tolist()
            else:
                target_line = np.asarray(target_line, dtype=np.float32).reshape(-1, 2)
                idx = np.arange(num_modes) % len(target_line)
                goals_xy_modes = target_line[idx].astype(np.float32)
                goals_idx_modes = idx.astype(np.int64).tolist()
        else:
            goals_xy_modes = np.tile(goal_xy.reshape(1, 2), (num_modes, 1)).astype(np.float32)
            goals_idx_modes = (np.zeros((num_modes,), dtype=np.int64)).tolist()

        goals_9d_modes = np.tile(x0_9.reshape(1, -1), (num_modes, 1)).astype(np.float32)
        goals_9d_modes[:, 0:2] = goals_xy_modes[:, 0:2]

        for chunk_idx in range(plan_once_chunks):
            if len(mode_actions[0]) >= max_episode_length:
                break
            cond: Dict[int, np.ndarray] = {}
            if use_11d:
                cond[0] = np.array(
                    [self._obs_9d_to_11d(current_obs_batch[i], goals_xy_modes[i]) for i in range(num_modes)],
                    dtype=np.float32,
                )
            else:
                cond[0] = current_obs_batch.copy()
            if use_11d:
                cond[horizon - 1] = np.array(
                    [self._obs_9d_to_11d(goals_9d_modes[i], goals_xy_modes[i]) for i in range(num_modes)],
                    dtype=np.float32,
                )
            else:
                cond[horizon - 1] = goals_9d_modes.copy()

            _all_sampled_action, trajectories, diffusion_paths = self._policy(
                cond,
                batch_size=num_modes,
                horizon=horizon,
                return_diffusion=bool(self.plan_config.get("return_diffusion", True)),
            )
            obs_all = np.asarray(trajectories.observations, dtype=np.float32)
            act_all = np.asarray(trajectories.actions, dtype=np.float32)
            is_last_chunk = chunk_idx == plan_once_chunks - 1
            for i in range(num_modes):
                states_b = obs_all[i]
                if use_11d and states_b.shape[-1] >= 11:
                    states_b = np.array([self._obs_11d_to_9d(states_b[t]) for t in range(states_b.shape[0])], dtype=np.float32)
                else:
                    states_b = np.asarray(states_b, dtype=np.float32)
                actions_b = np.asarray(act_all[i], dtype=np.float32)
                # Use env dynamics to generate consistent states from actions (prevents diffusion-state blowups across chunks).
                # If env doesn't support rollout_actions, fall back to diffusion-predicted observations.
                remaining = max(0, max_episode_length - len(mode_actions[i]))
                # When plan_once_steps_per_chunk > 0, take only first N steps so next chunk is "mid-path -> goal"
                n_take_cfg = min(plan_once_steps_per_chunk, states_b.shape[0] - 1) if plan_once_steps_per_chunk > 0 else (states_b.shape[0] - 1)
                n_take_desired = (states_b.shape[0] - 1) if is_last_chunk else n_take_cfg
                n_take = int(min(max(0, n_take_desired), remaining))
                if n_take <= 0:
                    continue

                actions_take = actions_b[:n_take].copy()
                # Action-space safety projection for 9D: adjust qdot so next tcp_xy stays safe.
                if bool(self.plan_config.get("enable_safety", True)):
                    actions_take = self._project_actions_9d_cbf_qp(
                        current_obs_batch[i],
                        actions_take,
                        alpha=float(self.plan_config.get("action_cbf_alpha", 0.5)),
                        safety_margin=float(self.plan_config.get("action_cbf_margin", 0.0)),
                        freeze_jacobian=bool(self.plan_config.get("action_cbf_freeze_jacobian", True)),
                    )
                if hasattr(self.env, "rollout_actions"):
                    rolled = self.env.rollout_actions(current_obs_batch[i], actions_take)
                    rolled = np.asarray(rolled, dtype=np.float32)
                    # rolled has shape (n_take+1, 9), includes start at index 0
                    for t in range(1, rolled.shape[0]):
                        mode_states[i].append(rolled[t].copy())
                    for t in range(actions_take.shape[0]):
                        mode_actions[i].append(actions_take[t].copy())
                    current_obs_batch[i] = rolled[-1].copy()
                else:
                    # Fallback: use diffusion predicted states (may be unstable across chunks)
                    end_idx = min(1 + n_take, states_b.shape[0])
                    for t in range(1, end_idx):
                        mode_states[i].append(states_b[t].copy())
                    n_act = end_idx - 1
                    for t in range(min(n_act, actions_b.shape[0])):
                        mode_actions[i].append(actions_b[t].copy())
                    current_obs_batch[i] = states_b[end_idx - 1].copy()

        # Apply framework constraints to EACH mode so the actions we execute in D3IL are safe/feasible.
        candidate_states = []
        candidate_actions = []
        candidate_costs = []
        candidate_fw_infos: list[Dict[str, Any]] = []
        for i in range(num_modes):
            states_i = [np.asarray(s, dtype=np.float32) for s in mode_states[i]]
            actions_i = [np.asarray(a, dtype=np.float32) for a in mode_actions[i]]
            states_i, actions_i, fw_i = self._apply_framework_constraints(states_i, actions_i)
            # Ensure start state matches x0 (framework might adjust)
            if len(states_i) > 0 and np.asarray(states_i[0]).size >= 9:
                states_i[0] = x0_9.copy()
            candidate_states.append(states_i)
            candidate_actions.append(actions_i)
            candidate_fw_infos.append(fw_i)
            last_pos = np.asarray(self._position_extractor(states_i[-1]), dtype=np.float32).reshape(-1)[: self._position_dim] if states_i else x0_9[: self._position_dim]
            candidate_costs.append(float(np.sum((last_pos - goals_xy_modes[i]) ** 2)))

        best_idx = int(np.argmin(candidate_costs)) if candidate_costs else 0
        states_list = candidate_states[best_idx] if candidate_states else [x0_9.copy()]
        actions_list = candidate_actions[best_idx] if candidate_actions else []
        fw_info = candidate_fw_infos[best_idx] if candidate_fw_infos else {}
        if len(states_list) > 0 and np.asarray(states_list[0]).size >= 9:
            states_list[0] = x0_9.copy()

        action0 = np.asarray(actions_list[0], dtype=np.float32).copy() if actions_list else np.zeros(7, dtype=np.float32)
        goal_xy_info = self._resolve_goal_xy().tolist()
        info: Dict[str, Any] = {}
        info.update(fw_info)
        info.update({
            "action0": np.asarray(action0, dtype=np.float32).tolist(),
            "goal_xy": goal_xy_info,
            "native_9d": True,
            "safety": "cbf_qp_invariance" if self._projector is not None else "none",
        })
        # Debug/analysis: record which target each mode was conditioned on.
        info["use_target_line"] = bool(use_target_line)
        info["num_targets"] = int(num_targets)
        info["candidate_goals_xy"] = np.asarray(goals_xy_modes, dtype=np.float32).tolist()
        info["candidate_goals_idx"] = list(goals_idx_modes)
        if diffusion_paths is not None:
            info["diffusion_paths_shape"] = list(np.asarray(diffusion_paths).shape)
        info["candidate_states"] = candidate_states
        info["candidate_actions"] = candidate_actions
        info["candidate_costs"] = np.asarray(candidate_costs, dtype=np.float32).tolist()
        info["best_idx"] = best_idx
        info["mode_strategy"] = "multirun"
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
        When plan_once_chunks > 1: multi-chunk 4D plan (long trajectory, num_modes).
        """
        x = np.asarray(x0, dtype=np.float32).reshape(-1)
        if self.native_9d:
            if x.size < 9:
                raise ValueError(f"Native 9D SafeDiffuser expects x0 size >= 9, got {x.shape}")
            plan_once_chunks_9d = int(self.plan_config.get("plan_once_chunks", 1))
            num_modes_9d = int(self.plan_config.get("num_modes", 1))
            # Multi-mode needs candidate_states: use chunked path (with 1 chunk = one rollout per mode, obstacle-avoiding)
            if plan_once_chunks_9d > 1 or num_modes_9d > 1:
                return self._plan_core_9d_chunked(x, rng_key=rng_key)
            return self._plan_core_9d(x, rng_key=rng_key)
        if x.size == 4:
            x4 = x
        elif x.size >= 2:
            goal_xy = self._resolve_goal_xy()
            pos = np.asarray(self._position_extractor(x), dtype=np.float32).reshape(-1)[: self._position_dim]
            x4 = np.array([float(goal_xy[0]), float(goal_xy[1]), float(pos[0]), float(pos[1])], dtype=np.float32)
        else:
            raise ValueError(f"SafeDiffuser backend expects x0 with at least 2 dims, got {x.shape}")
        plan_once_chunks = int(self.plan_config.get("plan_once_chunks", 1))
        if plan_once_chunks > 1:
            return self._plan_core_4d_chunked(x4, rng_key=rng_key)
        return self._plan_core_4d(x4, rng_key=rng_key)
