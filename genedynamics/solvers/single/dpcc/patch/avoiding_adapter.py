from __future__ import annotations

from copy import copy
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from genedynamics.core.task_spec import legacy_extract_position
from genedynamics.solvers.single.dpcc.patch.constraints_helpers import (
    formulate_bounds_constraints,
    formulate_dynamics_constraints,
    formulate_halfspace_constraints,
)


class AvoidingDPCCAdapter:
    """
    Adapter for D3IL Avoiding environment to DPCC-style observations/actions.
    """

    def __init__(
        self,
        env,
        config: Dict[str, Any],
        indices: Dict[str, Dict[str, int]],
        position_extractor=None,
    ):
        self.env = env
        self.config = config
        self.indices = indices
        self._position_extractor = position_extractor or legacy_extract_position

    def reset(self, seed: int | None = None):
        reset_out = self.env.reset(rng=None if seed is None else seed)
        if isinstance(reset_out, tuple):
            obs, _info = reset_out
        else:
            obs = reset_out
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        robot_state = np.asarray(self.env.robot_state(), dtype=np.float32).reshape(-1)
        action = np.asarray(self._position_extractor(robot_state), dtype=np.float32).reshape(-1)
        fixed_z = robot_state[2:]
        return obs, action, fixed_z

    def step(self, action, obs, fixed_z):
        # D3ILAvoidingEnv expects delta action (dx, dy); state arg unused.
        next_state, rew, terminated, info = self.env.step(None, np.asarray(action, dtype=np.float32))
        success = False
        if isinstance(info, dict) and "success" in info:
            success = bool(info["success"])
        elif isinstance(info, (tuple, list)) and len(info) > 1:
            success = bool(info[1])
        obs_next = np.asarray(next_state, dtype=np.float32).reshape(-1)
        return obs_next, success, terminated, info

    def get_indices(self):
        return self.indices.get("observations", {}), self.indices.get("actions", {})

    def build_constraint_variants(
        self,
        exp: str,
        robot_name: str,
        halfspace_variant: str | None,
        trajectory_dim: int,
        action_dim: int,
        act_obs_indices: Dict[str, int],
        runtime_obstacles=None,
        runtime_obstacle_config: Dict[str, Any] | None = None,
        align_with_framework: bool = True,
        disable_halfspace_when_aligned: bool = True,
    ):
        constraint_types = list(
            self._resolve_config("constraint_types", exp=exp, robot_name=robot_name, default=[])
        )
        polytopic_constraints, obstacle_constraints = self._select_avoiding_constraints(
            exp=exp, halfspace_variant=halfspace_variant
        )
        runtime_constraints = []
        if align_with_framework:
            runtime_constraints = self._runtime_obstacle_constraints(
                runtime_obstacles, runtime_obstacle_config
            )
            if runtime_constraints:
                obstacle_constraints = runtime_constraints
                if "obstacles" not in constraint_types:
                    constraint_types.append("obstacles")
                if disable_halfspace_when_aligned and "halfspace" in constraint_types:
                    constraint_types = [c for c in constraint_types if c != "halfspace"]
        bounds = self._resolve_config("bounds", exp=exp, robot_name=robot_name, default=[])
        enlarge_constraints = self._resolve_config(
            "enlarge_constraints", exp=exp, robot_name=robot_name, default=0.0
        )
        constraint_list = []
        constraint_list_tightened = []
        constraint_list_polytopic_not_tightened = []

        if "halfspace" in constraint_types:
            for constraint in polytopic_constraints:
                constraint_list.append(
                    (
                        "ineq",
                        formulate_halfspace_constraints(
                            constraint, 0, trajectory_dim, act_obs_indices
                        ),
                    )
                )
                constraint_list_tightened.append(
                    (
                        "ineq",
                        formulate_halfspace_constraints(
                            constraint, enlarge_constraints, trajectory_dim, act_obs_indices
                        ),
                    )
                )
                constraint_list_polytopic_not_tightened.append(
                    (
                        "ineq",
                        formulate_halfspace_constraints(
                            constraint, 0, trajectory_dim, act_obs_indices
                        ),
                    )
                )

        if "bounds" in constraint_types:
            lower_bound, upper_bound = formulate_bounds_constraints(
                constraint_types, bounds, trajectory_dim, act_obs_indices
            )
            constraint_list.extend([["lb", lower_bound], ["ub", upper_bound]])
            constraint_list_tightened.extend([["lb", lower_bound], ["ub", upper_bound]])
        else:
            lower_bound, upper_bound = None, None

        if "obstacles" in constraint_types:
            for constr in obstacle_constraints:
                constraint_list.append(
                    [
                        constr["type"],
                        [
                            act_obs_indices[constr["dimensions"][0]],
                            act_obs_indices[constr["dimensions"][1]],
                        ],
                        constr["center"],
                        constr["radius"],
                    ]
                )
                constraint_list_tightened.append(
                    [
                        constr["type"],
                        [
                            act_obs_indices[constr["dimensions"][0]],
                            act_obs_indices[constr["dimensions"][1]],
                        ],
                        constr["center"],
                        constr["radius"] + enlarge_constraints,
                    ]
                )

        constraint_list_without_prior = copy(constraint_list)
        constraint_list_without_prior_tightened = copy(constraint_list_tightened)

        dynamics_constraints = []
        if "dynamics" in constraint_types:
            dynamics_constraints = formulate_dynamics_constraints(exp, act_obs_indices, action_dim)
        for constraint in dynamics_constraints:
            constraint_list.append(constraint)
            constraint_list_tightened.append(constraint)

        return {
            "constraint_list": constraint_list,
            "constraint_list_tightened": constraint_list_tightened,
            "constraint_list_without_prior": constraint_list_without_prior,
            "constraint_list_without_prior_tightened": constraint_list_without_prior_tightened,
            "constraint_list_polytopic_not_tightened": constraint_list_polytopic_not_tightened,
            "constraint_types": constraint_types,
            "bounds": bounds,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "obstacle_constraints": obstacle_constraints,
            "obstacle_constraints_source": "framework" if runtime_constraints else "dpcc_config",
            "enlarge_constraints": enlarge_constraints,
            "polytopic_constraints": polytopic_constraints,
            "halfspace_variant": halfspace_variant,
        }

    def _resolve_config(self, key: str, *, exp: str, robot_name: str, default=None):
        if key not in self.config:
            return default
        value = self.config[key]
        if isinstance(value, dict):
            # DPCC config convention:
            # - dt/enlarge_constraints are keyed by robot_name (e.g. "avoiding")
            # - most other entries are keyed by exp (e.g. "avoiding-d3il")
            if key in {"dt", "enlarge_constraints"}:
                return value.get(robot_name, default)
            for exp_key in self._exp_candidates(exp):
                if exp_key in value:
                    return value[exp_key]
            return default
        return value

    def _exp_candidates(self, exp: str):
        candidates = [exp]
        if exp.endswith("-9d"):
            candidates.append(exp[:-3])  # fallback: avoiding-d3il-9d -> avoiding-d3il
        return candidates

    def _get_halfspace_list(self, exp: str):
        all_constraints = self.config.get("halfspace_constraints", {})
        if isinstance(all_constraints, dict):
            for exp_key in self._exp_candidates(exp):
                if exp_key in all_constraints:
                    return all_constraints[exp_key]
            return []
        return all_constraints or []

    def _get_obstacle_list(self, exp: str):
        all_constraints = self.config.get("obstacle_constraints", {})
        if isinstance(all_constraints, dict):
            for exp_key in self._exp_candidates(exp):
                if exp_key in all_constraints:
                    return all_constraints[exp_key]
            return []
        return all_constraints or []

    def _select_avoiding_constraints(
        self, *, exp: str, halfspace_variant: str | None
    ) -> Tuple[list, list]:
        """
        Match DPCC `dpcc/scripts/eval.py` selection for avoiding:
          - top-left-hard:  halfspace[0], obstacle[3]
          - top-right-hard: halfspace[1], obstacle[4]
          - both-hard:      halfspace[2] & halfspace[3], obstacle[5]
        If halfspace_variant is None, fall back to "all" (use full lists).
        """
        polytopic_all = self._get_halfspace_list(exp)
        obstacles_all = self._get_obstacle_list(exp)

        if halfspace_variant is None:
            return polytopic_all, obstacles_all

        if halfspace_variant == "top-left-hard":
            return [polytopic_all[0]], [obstacles_all[3]]
        if halfspace_variant == "top-right-hard":
            return [polytopic_all[1]], [obstacles_all[4]]
        if halfspace_variant == "both-hard":
            return [polytopic_all[2], polytopic_all[3]], [obstacles_all[5]]

        # Default: no special slicing
        return polytopic_all, obstacles_all

    def _runtime_obstacle_constraints(self, obstacles, obstacle_config: Dict[str, Any] | None) -> list:
        """
        Convert framework obstacles to DPCC sphere constraints.
        """
        if obstacles is None:
            return []
        robot_radius = 0.0
        if isinstance(obstacle_config, dict):
            robot_radius = float(obstacle_config.get("robot_radius", 0.0) or 0.0)

        obstacle_list = self._iter_obstacles(obstacles)
        runtime_constraints = []
        for obs in obstacle_list:
            center = getattr(obs, "center", None)
            radius = getattr(obs, "radius", None)
            if center is None or radius is None:
                continue
            center_arr = np.asarray(center, dtype=np.float32).reshape(-1)
            if center_arr.size < 2:
                continue
            runtime_constraints.append(
                {
                    "type": "sphere_outside",
                    "dimensions": ["x", "y"],
                    "center": [float(center_arr[0]), float(center_arr[1])],
                    "radius": float(radius) + robot_radius,
                }
            )
        return runtime_constraints

    def _iter_obstacles(self, obstacles) -> Iterable:
        if hasattr(obstacles, "obstacles"):
            return getattr(obstacles, "obstacles") or []
        if isinstance(obstacles, (list, tuple)):
            return obstacles
        return []
