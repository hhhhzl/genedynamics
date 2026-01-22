from __future__ import annotations

from copy import copy
from typing import Any, Dict, Tuple

import numpy as np

from enerdynamics.solvers.single.dpcc.patch.constraints_helpers import (
    formulate_bounds_constraints,
    formulate_dynamics_constraints,
    formulate_halfspace_constraints,
)


class AvoidingDPCCAdapter:
    """
    Adapter for D3IL Avoiding environment to DPCC-style observations/actions.
    """

    def __init__(self, env, config: Dict[str, Any], indices: Dict[str, Dict[str, int]]):
        self.env = env
        self.config = config
        self.indices = indices

    def reset(self, seed: int | None = None):
        obs = self.env.reset() if seed is None else self.env.reset(seed=seed)
        action = self.env.robot_state()[:2]
        fixed_z = self.env.robot_state()[2:]
        obs = np.concatenate((action[:2], obs))
        return obs, action, fixed_z

    def step(self, action, obs, fixed_z):
        next_pos_des = action + obs[:2]
        obs, rew, terminated, info = self.env.step(
            np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0)
        )
        success = info[1] if isinstance(info, (tuple, list)) and len(info) > 1 else info
        obs = np.concatenate((next_pos_des[:2], obs))
        return obs, success, terminated, info

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
    ):
        constraint_types = self._resolve_config("constraint_types", exp=exp, robot_name=robot_name, default=[])
        polytopic_constraints, obstacle_constraints = self._select_avoiding_constraints(
            exp=exp, halfspace_variant=halfspace_variant
        )
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
            return value.get(exp, default)
        return value

    def _get_halfspace_list(self, exp: str):
        all_constraints = self.config.get("halfspace_constraints", {})
        if isinstance(all_constraints, dict):
            return all_constraints.get(exp, [])
        return all_constraints or []

    def _get_obstacle_list(self, exp: str):
        all_constraints = self.config.get("obstacle_constraints", {})
        if isinstance(all_constraints, dict):
            return all_constraints.get(exp, [])
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

