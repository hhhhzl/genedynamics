from __future__ import annotations

import numpy as np


def formulate_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, act_obs_indices):
    m = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
    n = [-1, 1 / m] / np.linalg.norm([-1, 1 / m])
    if (m > 0 and constraint[2] == "below") or (m < 0 and constraint[2] == "above"):
        n *= -1
    points_enlarged = [
        constraint[0] + enlarge_constraints * n,
        constraint[1] + enlarge_constraints * n,
    ]
    d = points_enlarged[0][1] - m * points_enlarged[0][0]
    c_row = np.zeros(trajectory_dim)
    if constraint[2] == "below":
        c_row[act_obs_indices["x"]] = -m
        c_row[act_obs_indices["y"]] = 1
    elif constraint[2] == "above":
        c_row[act_obs_indices["x"]] = m
        c_row[act_obs_indices["y"]] = -1
        d *= -1
    return c_row, d


def formulate_bounds_constraints(constraint_types, bounds, trajectory_dim, act_obs_indices):
    lower_bound = -np.inf * np.ones(trajectory_dim)
    upper_bound = np.inf * np.ones(trajectory_dim)
    if "bounds" in constraint_types:
        for bound in bounds:
            for dim_idx, dim in enumerate(bound["dimensions"]):
                if bound["type"] == "lower" and dim in act_obs_indices:
                    lower_bound[act_obs_indices[dim]] = bound["values"][dim_idx]
                elif bound["type"] == "upper" and dim in act_obs_indices:
                    upper_bound[act_obs_indices[dim]] = bound["values"][dim_idx]
    return lower_bound, upper_bound


def formulate_dynamics_constraints(exp, act_obs_indices, action_dim):
    dynamic_constraints = []
    if "pointmaze" in exp:
        dynamic_constraints = [
            ("deriv", np.array([act_obs_indices["x"], act_obs_indices["vx"]])),
            ("deriv", np.array([act_obs_indices["y"], act_obs_indices["vy"]])),
        ]
    if "antmaze" in exp:
        dynamic_constraints = [
            ("deriv", np.array([act_obs_indices["x"], act_obs_indices["vx"]])),
            ("deriv", np.array([act_obs_indices["y"], act_obs_indices["vy"]])),
            ("deriv", np.array([act_obs_indices["z"], act_obs_indices["vz"]])),
        ]
    if "avoiding" in exp and action_dim > 0:
        dynamic_constraints = [
            ("deriv", np.array([act_obs_indices["x"], act_obs_indices["vx"]])),
            ("deriv", np.array([act_obs_indices["y"], act_obs_indices["vy"]])),
            ("deriv", np.array([act_obs_indices["x_des"], act_obs_indices["vx"]])),
            ("deriv", np.array([act_obs_indices["y_des"], act_obs_indices["vy"]])),
        ]
    return dynamic_constraints

