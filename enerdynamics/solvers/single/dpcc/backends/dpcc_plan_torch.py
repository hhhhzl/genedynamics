from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from enerdynamics.core.types import Trajectory
from enerdynamics.solvers.single.dpcc.patch.avoiding_adapter import AvoidingDPCCAdapter
from enerdynamics.solvers.single.dpcc.patch.projector import Projector


class DPCCBackendTorch:
    def __init__(
        self,
        *,
        env: Any,
        diffusion: Any,
        normalizer: Any,
        plan_config: Dict[str, Any],
        constraint_config: Dict[str, Any],
        indices: Dict[str, Dict[str, int]],
        device: str = "cuda",
        seed: int = 0,
        adapter: AvoidingDPCCAdapter | None = None,
    ):
        self.env = env
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.plan_config = plan_config
        self.constraint_config = constraint_config
        self.indices = indices
        self.device = device
        self.seed = seed
        self.adapter = adapter or AvoidingDPCCAdapter(env, constraint_config, indices)

        self._projector = None
        self._policy = None
        self._constraints_cache = None
        self._stepper = None

    def _build_projector_and_policy(self):
        if self._projector is not None and self._policy is not None:
            return

        variant = self.plan_config.get("variant", "dpcc")
        exp = self.plan_config.get("exp", self.constraint_config.get("exp", "avoiding-d3il"))
        robot_name = exp.split("-")[0]

        halfspace_variant = self.plan_config.get("halfspace_variant")
        if halfspace_variant is None and "avoiding" in exp:
            hv = self.constraint_config.get("avoiding_halfspace_variants") or []
            halfspace_variant = hv[0] if len(hv) > 0 else None
        align_with_framework = bool(
            self.plan_config.get("align_constraints_with_framework", True)
        )
        disable_halfspace_when_aligned = bool(
            self.plan_config.get("disable_halfspace_when_aligned", True)
        )

        # Match DPCC eval.py: parse gradient + dt multipliers from variant string.
        gradient = True if "gradient" in str(variant) else False
        dt_multiplier = float(self.plan_config.get("dt_multiplier", 1.0))
        if "dt0p25" in str(variant):
            dt_multiplier = 0.25
        elif "dt0p5" in str(variant):
            dt_multiplier = 0.5
        elif "dt2p0" in str(variant):
            dt_multiplier = 2.0
        elif "dt4p0" in str(variant):
            dt_multiplier = 4.0

        trajectory_dim = self.diffusion.transition_dim - self.diffusion.goal_dim
        action_dim = self.diffusion.action_dim

        obs_indices = self.indices.get("observations", {})
        act_indices = self.indices.get("actions", {})
        obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
        act_obs_indices = {**act_indices, **obs_indices_updated}

        constraints_info = self.adapter.build_constraint_variants(
            exp=exp,
            robot_name=robot_name,
            halfspace_variant=halfspace_variant,
            trajectory_dim=trajectory_dim,
            action_dim=action_dim,
            act_obs_indices=act_obs_indices,
            runtime_obstacles=self.plan_config.get("obstacles"),
            runtime_obstacle_config=self.plan_config.get("obstacle_config", {}),
            align_with_framework=align_with_framework,
            disable_halfspace_when_aligned=disable_halfspace_when_aligned,
        )
        if "model_free" in variant and "tightened" in variant:
            constraints = constraints_info["constraint_list_without_prior_tightened"]
        elif "model_free" in variant and "tightened" not in variant:
            constraints = constraints_info["constraint_list_without_prior"]
        elif "model_free" not in variant and "tightened" in variant:
            constraints = constraints_info["constraint_list_tightened"]
        else:
            constraints = constraints_info["constraint_list"]

        self._constraints_cache = constraints_info

        # DPCC config convention: dt is keyed by robot_name (e.g. "avoiding")
        dt_default = None
        dt_cfg = self.constraint_config.get("dt")
        if isinstance(dt_cfg, dict):
            dt_default = dt_cfg.get(robot_name)
        dt = self.plan_config.get("dt", dt_default if dt_default is not None else 0.1)
        delta_t = float(dt) * float(dt_multiplier)

        projector = Projector(
            horizon=self.plan_config.get("horizon", self.diffusion.horizon),
            transition_dim=trajectory_dim,
            action_dim=action_dim,
            goal_dim=self.diffusion.goal_dim,
            constraint_list=constraints,
            normalizer=self.normalizer,
            gradient=gradient,
            gradient_weights=self.plan_config.get("gradient_weights", [1, 0.5, 2]),
            variant="states_actions",
            dt=delta_t,
            cost_dims=self.plan_config.get("cost_dims"),
            device=self.device,
            solver=self.plan_config.get("solver", "scipy"),
        )
        
        # Match DPCC eval.py: disable projector only for "diffuser" baseline.
        if variant == "diffuser":
            projector = None

        trajectory_selection = self.plan_config.get("trajectory_selection", "random")
        if "dpcc-t" in variant:
            trajectory_selection = "temporal_consistency"
        elif "dpcc-c" in variant:
            trajectory_selection = "minimum_projection_cost"

        # Route denoising loop through solver-side DPCC stepper.
        if self._stepper is None:
            from enerdynamics.solvers.single.dpcc.stepper import DPCCTorchStepper
            self._stepper = DPCCTorchStepper(self.diffusion)
            self.diffusion.p_sample = self._stepper.p_sample
            self.diffusion.p_sample_loop = self._stepper.p_sample_loop
            self.diffusion.grad_p_sample = self._stepper.grad_p_sample
            self.diffusion.grad_p_sample_loop = self._stepper.grad_p_sample_loop
            self.diffusion.grad_conditional_sample = self._stepper.grad_conditional_sample

        from diffuser.sampling.policies import Policy
        policy = Policy(
            model=self.diffusion,
            normalizer=self.normalizer,
            preprocess_fns=self.plan_config.get("preprocess_fns", []),
            test_ret=self.plan_config.get("test_ret", 0),
            projector=projector,
            trajectory_selection=trajectory_selection,
        )

        self._projector = projector
        self._policy = policy

    def _plan_once(self, x0: Any, horizon: int, rng_key: Any | None) -> Dict[str, Any]:
        """Plan once (one or more diffusion+projection chunks) and return planned trajectory + multi-mode candidates."""
        x0 = np.asarray(x0, dtype=np.float32).reshape(-1)
        obs = x0.copy()
        if self.diffusion.observation_dim == 11 and obs.size == 9:
            target_xy = np.asarray(self.adapter.env.target, dtype=np.float32).reshape(-1)[:2]
            obs = np.concatenate([target_xy, obs], axis=0)

        num_modes = int(self.plan_config.get("num_modes", self.plan_config.get("batch_size", 4)))
        plan_once_chunks = int(self.plan_config.get("plan_once_chunks", 1))
        use_target_line = bool(self.plan_config.get("use_target_line", False))
        num_targets = int(self.plan_config.get("num_targets", 4))
        max_episode_length = int(self.plan_config.get("max_episode_length", 200))
        constraints = self._constraints_cache.get("constraint_list", []) if self._constraints_cache else []

        if plan_once_chunks <= 1:
            # Single chunk: one plan call, C candidates
            if use_target_line and self.diffusion.observation_dim == 11 and obs.size >= 11:
                from enerdynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
                target_line = get_d3il_target_line_positions(num_targets)
                obs_batch = np.tile(np.asarray(obs, dtype=np.float32).reshape(1, -1), (num_modes, 1))
                for i in range(num_modes):
                    obs_batch[i, :2] = target_line[i % len(target_line)]
                cond = {0: obs_batch}
                batch_cond = True
            else:
                cond = {0: obs}
                batch_cond = False
            out = self._policy(
                conditions=cond,
                batch_size=num_modes,
                horizon=horizon,
                disable_projection=False,
                constraints=constraints,
                return_infos=True,
                batch_conditions=batch_cond,
            )
            action, trajectories, infos = out
            observations = np.asarray(trajectories.observations, dtype=np.float32)
            actions = np.asarray(trajectories.actions, dtype=np.float32)
        else:
            # Batched chunked (EBMBD-aligned): num_modes independent long trajectories in one batched loop
            obs_dim = int(self.diffusion.observation_dim)
            x0_obs_chunk = np.asarray(obs[:obs_dim], dtype=np.float32) if obs.size >= obs_dim else np.asarray(obs, dtype=np.float32)
            # current_obs_batch: (num_modes, obs_dim) — one condition per mode
            current_obs_batch = np.tile(np.asarray(obs, dtype=np.float32).reshape(1, -1), (num_modes, 1))
            if use_target_line and obs_dim >= 11:
                from enerdynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
                target_line = get_d3il_target_line_positions(num_targets)
                for i in range(num_modes):
                    current_obs_batch[i, :2] = target_line[i % len(target_line)]
            # Per-mode trajectory lists: list of (T, obs_dim) states and (T, action_dim) actions
            mode_states = [[x0_obs_chunk.copy()] for _ in range(num_modes)]
            mode_actions = [[] for _ in range(num_modes)]
            cost_per_mode = np.zeros(num_modes, dtype=np.float64)
            for _ in range(plan_once_chunks):
                if len(mode_actions[0]) >= max_episode_length:
                    break
                out = self._policy(
                    conditions={0: current_obs_batch},
                    batch_size=num_modes,
                    horizon=horizon,
                    disable_projection=False,
                    constraints=constraints,
                    return_infos=True,
                    batch_conditions=True,
                )
                _action, trajectories, infos = out
                observations = np.asarray(trajectories.observations, dtype=np.float32)  # (num_modes, horizon, obs_dim)
                actions = np.asarray(trajectories.actions, dtype=np.float32)  # (num_modes, horizon, action_dim)
                for i in range(num_modes):
                    for t in range(observations.shape[1]):
                        mode_states[i].append(observations[i, t].copy())
                    for t in range(actions.shape[1]):
                        mode_actions[i].append(actions[i, t].copy())
                    current_obs_batch[i] = observations[i, -1]
                projection_costs = infos.get("projection_costs")
                if projection_costs is not None and isinstance(projection_costs, dict):
                    for cost_arr in projection_costs.values():
                        arr = np.asarray(cost_arr, dtype=np.float64).ravel()
                        if arr.size >= num_modes:
                            cost_per_mode += arr[:num_modes]
            # Build (num_modes, T, obs_dim) and (num_modes, T, action_dim) for candidate_*; ensure first state is x0
            observations = np.zeros((num_modes, len(mode_states[0]), obs_dim), dtype=np.float32)
            for i in range(num_modes):
                for t, s in enumerate(mode_states[i]):
                    observations[i, t] = np.asarray(s, dtype=np.float32)[:obs_dim]
            act_dim = np.asarray(mode_actions[0][0]).ravel().size if mode_actions[0] else 1
            actions = np.zeros((num_modes, len(mode_actions[0]), act_dim), dtype=np.float32)
            for i in range(num_modes):
                for t, a in enumerate(mode_actions[i]):
                    actions[i, t] = np.asarray(a, dtype=np.float32).ravel()[:act_dim]
            # Overwrite first state of every candidate with x0 for viz
            for i in range(num_modes):
                observations[i, 0] = x0_obs_chunk
            infos = {"projection_costs": {0: cost_per_mode}}

        C = observations.shape[0]
        obs_dim = observations.shape[2] if observations.ndim >= 3 else observations.shape[1]
        # Ensure first state of every candidate is exactly x0 (for correct start in viz)
        x0_obs = np.asarray(obs[:obs_dim], dtype=np.float32) if obs.size >= obs_dim else np.asarray(obs, dtype=np.float32)

        candidate_states = []
        candidate_actions = []
        for i in range(C):
            # states = actions + 1: x0 then planned steps; add duplicate last state only when H == num_actions (single-chunk)
            H = observations.shape[1]
            states_i = [x0_obs.copy()] + [observations[i, t] for t in range(1, H)]
            if actions.shape[1] == H:
                states_i = states_i + [np.asarray(observations[i, H - 1], dtype=np.float32)]
            acts_i = [actions[i, t] for t in range(actions.shape[1])]
            candidate_states.append(states_i)
            candidate_actions.append(acts_i)

        # Total projection cost per candidate (sum over timesteps)
        projection_costs = infos.get("projection_costs")
        if projection_costs is not None and isinstance(projection_costs, dict):
            cost_per_sample = np.zeros(C, dtype=np.float64)
            for _timestep, cost_arr in projection_costs.items():
                arr = np.asarray(cost_arr, dtype=np.float64).ravel()
                if arr.size >= C:
                    cost_per_sample += arr[:C]
            candidate_costs = cost_per_sample.astype(np.float32)
        else:
            candidate_costs = np.zeros(C, dtype=np.float32)

        best_idx = int(np.argmin(candidate_costs))
        states = candidate_states[best_idx]
        actions_list = candidate_actions[best_idx]

        info = {
            "candidate_states": candidate_states,
            "candidate_actions": candidate_actions,
            "candidate_costs": candidate_costs,
            "best_idx": best_idx,
            "mode_strategy": "multirun",
        }
        return {
            "states": states,
            "actions": actions_list,
            "initial_state": x0,
            "info": info,
            "candidate_states": candidate_states,
            "candidate_actions": candidate_actions,
            "candidate_costs": candidate_costs,
            "best_idx": best_idx,
        }

    def plan(self, x0: Any | None = None, rng_key: Any | None = None) -> Dict[str, Any]:
        self._build_projector_and_policy()

        execution = self.plan_config.get("execution", "mpc")
        if execution == "plan_once":
            horizon = int(self.plan_config.get("horizon", self.diffusion.horizon))
            x0_use = x0 if x0 is not None else self.adapter.reset(seed=self.seed)[0]
            return self._plan_once(x0_use, horizon, rng_key)

        max_episode_length = int(self.plan_config.get("max_episode_length", 200))
        batch_size = int(self.plan_config.get("batch_size", 8))
        horizon = int(self.plan_config.get("horizon", self.diffusion.horizon))
        n_trials = int(self.plan_config.get("n_trials", 1))
        disable_projection = bool(self.plan_config.get("disable_projection", False))
        save_samples_every = max(1, horizon // 2)

        exp = self.plan_config.get("exp", self.constraint_config.get("exp", "avoiding-d3il"))
        obs_indices = self.indices.get("observations", {})
        action_dim = self.diffusion.action_dim
        constraint_types = (self._constraints_cache or {}).get("constraint_types", [])
        polytopic_not_tight = (self._constraints_cache or {}).get(
            "constraint_list_polytopic_not_tightened", []
        )
        obstacle_constraints = (self._constraints_cache or {}).get("obstacle_constraints", [])
        lower_bound = (self._constraints_cache or {}).get("lower_bound", None)
        upper_bound = (self._constraints_cache or {}).get("upper_bound", None)

        # Match dpcc/scripts/eval.py arrays
        n_success = np.zeros(n_trials, dtype=np.float32)
        n_success_and_constraints = np.zeros(n_trials, dtype=np.float32)
        n_steps = np.zeros(n_trials, dtype=np.float32)
        n_violations = np.zeros(n_trials, dtype=np.float32)
        total_violations = np.zeros(n_trials, dtype=np.float32)
        avg_time = np.zeros(n_trials, dtype=np.float32)
        collision_free_completed = np.ones(n_trials, dtype=np.float32)
        pos_tracking_errors = np.zeros((n_trials, max_episode_length - 1), dtype=np.float32)

        sampled_trajectories_all = []
        obs_buffers_all = []
        action_buffers_all = []
        fixed_z_all = []
        initial_q_all = []

        # Run trials (match eval.py loop structure)
        for i in range(n_trials):
            try:
                import torch

                torch.manual_seed(i)
            except Exception:
                pass

            # Match dpcc/scripts/eval.py for avoiding: env.reset() without a seed per trial.
            obs, action, fixed_z = self.adapter.reset(seed=None)
            obs0 = np.asarray(obs, dtype=np.float32)
            fixed_z_all.append(np.asarray(fixed_z, dtype=np.float32).reshape(-1))
            initial_q_all.append(self._get_current_robot_q())

            # DPCC eval buffers (do NOT include initial obs0)
            obs_buffer_dpcc = []
            action_buffer = []
            sampled_trajectories = []
            desired_next_pos = None

            for t in range(max_episode_length):
                # 9D with target: env returns 9D state; policy expects 11D [x_des, y_des, x, y, q1..q7]. Inject target.
                if self.diffusion.observation_dim == 11 and obs.size == 9:
                    target_xy = np.asarray(self.adapter.env.target, dtype=np.float32).reshape(-1)[:2]
                    obs = np.concatenate([target_xy, obs], axis=0)
                # --- Safety violation checks (match dpcc/scripts/eval.py) ---
                violated_this_timestep = 0
                if "halfspace" in constraint_types:
                    for constraint in polytopic_not_tight:
                        if constraint[0] == "ineq":
                            c, d = constraint[1]
                            obs_to_check = (
                                obs[:- self.diffusion.goal_dim]
                                if self.diffusion.goal_dim > 0
                                else obs
                            )
                            if obs_to_check @ c[action_dim:] >= d:
                                violated_this_timestep = 1
                                total_violations[i] += float(obs_to_check @ c[action_dim:] - d)
                                collision_free_completed[i] = 0.0

                if "obstacles" in constraint_types:
                    for constr in obstacle_constraints:
                        xy = obs[[obs_indices["x"], obs_indices["y"]]]
                        if np.linalg.norm(xy - constr["center"]) < constr["radius"]:
                            violated_this_timestep = 1
                            total_violations[i] += float(
                                constr["radius"] - np.linalg.norm(xy - constr["center"])
                            )
                            collision_free_completed[i] = 0.0

                if (
                    t > 0
                    and "bounds" in constraint_types
                    and lower_bound is not None
                    and upper_bound is not None
                ):
                    act_obs = np.concatenate((action, obs)) if action_dim > 0 else obs
                    total_violations[i] += float(
                        np.sum(np.maximum(0, act_obs - upper_bound))
                        + np.sum(np.maximum(0, lower_bound - act_obs))
                    )

                n_violations[i] += violated_this_timestep

                # Calculate action
                start = time.time()

                action, samples = self._policy(
                    conditions={0: obs},
                    batch_size=batch_size,
                    horizon=horizon,
                    disable_projection=disable_projection,
                )
                avg_time[i] += time.time() - start

                # 9D eval: clip qdot to control_limit before env.step (no projector in diffuser variant)
                if "9d" in str(exp) and action_dim == 7:
                    limit = float(getattr(self.adapter.env, "control_limit", 1.5))
                    action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -limit, limit)

                # Step environment
                obs, success, terminated, info = self.adapter.step(action, obs, fixed_z)
                # Tracking error (match eval.py)
                if t >= 1 and desired_next_pos is not None:
                    pos_tracking_errors[i, t - 1] = float(
                        np.linalg.norm(
                            obs[obs_indices["x"] : obs_indices["y"] + 1] - desired_next_pos
                        )
                    )
                desired_next_pos = samples.observations[0, 1, [obs_indices["x"], obs_indices["y"]]]

                if t % save_samples_every == 0:
                    sampled_trajectories.append(samples.observations[:, :, :])

                # Store 9D env state for trajectory (framework expects 9D for 9d exp)
                obs_to_store = obs[2:11] if (obs.size == 11) else obs
                obs_buffer_dpcc.append(obs_to_store)
                action_buffer.append(action)

                if success:
                    n_success[i] = 1.0
                if (terminated or t == max_episode_length - 1) and (not success):
                    collision_free_completed[i] = 0.0

                if success or terminated or t == max_episode_length - 1:
                    n_steps[i] = float(t)
                    # Match dpcc/scripts/eval.py: avg_time[i] /= t (note: t can be 0)
                    denom = float(t)
                    avg_time[i] = avg_time[i] / denom if denom != 0.0 else avg_time[i] / 0.0
                    if success and collision_free_completed[i]:
                        n_success_and_constraints[i] = 1.0
                    break

            sampled_trajectories_all.append(sampled_trajectories)
            # Keep both:
            # - dpcc-style buffer (no initial)
            # - framework trajectory states (include initial to satisfy states = actions + 1)
            obs_buffers_all.append([obs0] + [np.asarray(s, dtype=np.float32) for s in obs_buffer_dpcc])
            action_buffers_all.append(action_buffer)

        # Return the first trial as the primary trajectory (framework expects a single trajectory)
        states = [np.asarray(s, dtype=np.float32) for s in obs_buffers_all[0]]
        actions = [np.asarray(a, dtype=np.float32) for a in action_buffers_all[0]]
        lifted_states_9d, lifted_actions_9d = self._lift_4d_to_9d_trajectory(
            states=states,
            initial_q=(initial_q_all[0] if len(initial_q_all) > 0 else None),
            fixed_z=(fixed_z_all[0] if len(fixed_z_all) > 0 else None),
            exp=exp,
        )

        info = {
            # Primary (trial-0) summary
            "success": bool(n_success[0] > 0),
            "steps": int(len(actions)),
            "avg_time": float(avg_time[0]),
            # DPCC-eval-aligned arrays
            "n_success": n_success,
            "n_success_and_constraints": n_success_and_constraints,
            "n_steps": n_steps,
            "n_violations": n_violations,
            "total_violations": total_violations,
            "avg_time_all": avg_time,
            "collision_free_completed": collision_free_completed,
            "pos_tracking_errors": pos_tracking_errors,
            "sampled_trajectories_all": sampled_trajectories_all,
            # DPCC-style state buffer (trial 0, WITHOUT initial obs0) for exact parity checks
            "states_dpcc_trial0": [np.asarray(s, dtype=np.float32) for s in obs_buffers_all[0][1:]],
            # For debugging parity
            "exp": exp,
            "variant": self.plan_config.get("variant", "dpcc"),
            "n_trials": n_trials,
            "horizon": horizon,
            "batch_size": batch_size,
            "max_episode_length": max_episode_length,
        }
        if lifted_states_9d is not None and lifted_actions_9d is not None:
            info["states_9d"] = lifted_states_9d
            info["actions_9d"] = lifted_actions_9d
            info["state_layout_9d"] = "[x, y, q1..q7]"
            info["action_layout_9d"] = "[qdot1..qdot7]"
        result = {"states": states, "actions": actions, "info": info}
        if lifted_states_9d is not None and lifted_actions_9d is not None:
            result["states_9d"] = lifted_states_9d
            result["actions_9d"] = lifted_actions_9d
        return result

    def sample_trajectories(
        self, x0: Any | None, n_samples: int, rng_key: Any | None = None
    ) -> List[Trajectory]:
        self._build_projector_and_policy()
        horizon = int(self.plan_config.get("horizon", self.diffusion.horizon))
        obs = x0 if x0 is not None else self.adapter.reset(seed=self.seed)[0]
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.diffusion.observation_dim == 11 and obs.size == 9:
            target_xy = np.asarray(self.adapter.env.target, dtype=np.float32).reshape(-1)[:2]
            obs = np.concatenate([target_xy, obs], axis=0)

        _, samples = self._policy(
            conditions={0: obs}, batch_size=n_samples, horizon=horizon, disable_projection=False
        )
        trajectories = []
        for i in range(n_samples):
            states = [np.asarray(s, dtype=np.float32) for s in samples.observations[i]]
            actions = [np.asarray(a, dtype=np.float32) for a in samples.actions[i]]
            if len(states) == len(actions):
                states.append(np.asarray(states[-1], dtype=np.float32))
            trajectories.append(Trajectory(states=states, actions=actions))
        return trajectories

    def _lift_4d_to_9d_trajectory(
        self,
        *,
        states: List[np.ndarray],
        initial_q: Optional[np.ndarray],
        fixed_z: Optional[np.ndarray],
        exp: str,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """
        Convert 4D avoiding trajectory [x_des, y_des, x, y] to 9D
        [x, y, q1..q7] by differential IK (Jacobian pseudo-inverse).
        """
        if "9d" in str(exp).lower() or len(states) < 2:
            return None, None
        obs_indices = self.indices.get("observations", {})
        if "x" not in obs_indices or "y" not in obs_indices:
            return None, None

        robot, _quat = self._get_robot_and_quat()
        if robot is None:
            return None, None
        if not hasattr(robot, "getForwardKinematics") or not hasattr(robot, "getJacobian"):
            return None, None

        if initial_q is None:
            return None, None
        q_cur = np.asarray(initial_q, dtype=np.float32).reshape(-1)[:7].copy()
        if q_cur.size != 7:
            return None, None

        # Keep IK integration consistent with 9D environment timing.
        dt = float(self.plan_config.get("lift_ik_dt", 0.035))
        dt = max(1e-6, dt)
        _ = fixed_z

        states_9d: List[np.ndarray] = []
        actions_9d: List[np.ndarray] = []
        for t in range(len(states)):
            s_t = np.asarray(states[t], dtype=np.float32).reshape(-1)
            xy_t = np.array([s_t[obs_indices["x"]], s_t[obs_indices["y"]]], dtype=np.float32)

            state_9d_t = np.concatenate([xy_t, q_cur.astype(np.float32)], axis=0)
            states_9d.append(state_9d_t)
            if t >= len(states) - 1:
                break

            s_next = np.asarray(states[t + 1], dtype=np.float32).reshape(-1)
            xy_next = np.array(
                [s_next[obs_indices["x"]], s_next[obs_indices["y"]]], dtype=np.float32
            )
            q_prev = q_cur.copy()
            try:
                q_solved = self._solve_ik_xy(
                    robot=robot, q_init=q_prev, xy_target=xy_next, max_iters=20, tol=2e-3
                )
                qdot = (q_solved - q_prev) / dt
                qdot = np.asarray(qdot, dtype=np.float32).reshape(-1)[:7]
                qdot = np.clip(qdot, -1.5, 1.5)
                q_next = q_prev + dt * qdot
                if hasattr(robot, "joint_pos_min") and hasattr(robot, "joint_pos_max"):
                    qmin = np.asarray(robot.joint_pos_min, dtype=np.float32).reshape(-1)[:7]
                    qmax = np.asarray(robot.joint_pos_max, dtype=np.float32).reshape(-1)[:7]
                    q_next = np.clip(q_next, qmin, qmax)
                q_cur = q_next.astype(np.float32)
            except Exception:
                # Keep continuity even if one IK step fails.
                qdot = np.zeros(7, dtype=np.float32)
                q_cur = q_prev

            actions_9d.append(np.asarray(qdot, dtype=np.float32))

        # ensure action length matches trajectory convention (N-1)
        if len(actions_9d) > max(0, len(states_9d) - 1):
            actions_9d = actions_9d[: len(states_9d) - 1]
        return states_9d, actions_9d

    def _solve_ik_xy(
        self,
        *,
        robot: Any,
        q_init: np.ndarray,
        xy_target: np.ndarray,
        max_iters: int = 20,
        tol: float = 2e-3,
    ) -> np.ndarray:
        """
        Damped least-squares IK in XY only.
        """
        q = np.asarray(q_init, dtype=np.float32).reshape(-1)[:7].copy()
        target = np.asarray(xy_target, dtype=np.float32).reshape(-1)[:2]
        lam = 1e-3

        for _ in range(max_iters):
            pos, _quat = robot.getForwardKinematics(q)
            ee_xy = np.asarray(pos, dtype=np.float32).reshape(-1)[:2]
            err = target - ee_xy
            if float(np.linalg.norm(err)) <= tol:
                break

            J = np.asarray(robot.getJacobian(q), dtype=np.float32)
            J_xy = J[:2, :7]
            JJt = J_xy @ J_xy.T
            step_xy = np.linalg.solve(JJt + lam * np.eye(2, dtype=np.float32), err)
            dq = J_xy.T @ step_xy
            q = q + dq.astype(np.float32)

            if hasattr(robot, "joint_pos_min") and hasattr(robot, "joint_pos_max"):
                qmin = np.asarray(robot.joint_pos_min, dtype=np.float32).reshape(-1)[:7]
                qmax = np.asarray(robot.joint_pos_max, dtype=np.float32).reshape(-1)[:7]
                q = np.clip(q, qmin, qmax)
        return q.astype(np.float32)

    def _get_current_robot_q(self) -> Optional[np.ndarray]:
        robot, _quat = self._get_robot_and_quat()
        if robot is None:
            return None
        try:
            robot.receiveState()
            q = np.asarray(robot.current_j_pos, dtype=np.float32).reshape(-1)[:7]
            if q.size != 7:
                return None
            return q.copy()
        except Exception:
            return None

    def _get_robot_and_quat(self) -> Tuple[Any, np.ndarray]:
        quat_default = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        inner = getattr(getattr(self.env, "_task_env", None), "_env", None)
        robot = getattr(inner, "robot", None) if inner is not None else None
        if robot is None:
            return None, quat_default
        quat = quat_default
        try:
            robot.receiveState()
            if hasattr(robot, "current_c_quat"):
                quat = np.asarray(robot.current_c_quat, dtype=np.float32).reshape(-1)[:4]
        except Exception:
            pass
        return robot, quat
