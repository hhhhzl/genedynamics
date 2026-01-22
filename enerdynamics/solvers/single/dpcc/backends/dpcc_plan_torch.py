from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from enerdynamics.core.types import Trajectory
from enerdynamics.solvers.single.dpcc.patch.avoiding_adapter import AvoidingDPCCAdapter
from enerdynamics.solvers.single.dpcc.patch.policy import Policy
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

    def plan(self, x0: Any | None = None, rng_key: Any | None = None) -> Dict[str, Any]:
        self._build_projector_and_policy()

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

            # DPCC eval buffers (do NOT include initial obs0)
            obs_buffer_dpcc = []
            action_buffer = []
            sampled_trajectories = []
            desired_next_pos = None

            for t in range(max_episode_length):
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

                obs_buffer_dpcc.append(obs)
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
        return {"states": states, "actions": actions, "info": info}

    def sample_trajectories(
        self, x0: Any | None, n_samples: int, rng_key: Any | None = None
    ) -> List[Trajectory]:
        self._build_projector_and_policy()
        horizon = int(self.plan_config.get("horizon", self.diffusion.horizon))
        obs = x0 if x0 is not None else self.adapter.reset(seed=self.seed)[0]

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

