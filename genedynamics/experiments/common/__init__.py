"""
Common utilities for experiments.

This package provides shared utilities for obstacle generation,
constraint creation, and other common functionality.
"""

from .constraints import (
    create_constraint_pipeline,
)
from .obstacle_generation import (
    generate_box2d_obstacles,
    check_obstacle_spacing,
    get_obstacle_radius,
    check_start_target_clearance,
    has_free_space_path,
    compute_nonconvexity_score_sdf,
)

__all__ = [
    "create_constraint_pipeline",
    "generate_box2d_obstacles",
    "check_obstacle_spacing",
    "get_obstacle_radius",
    "check_start_target_clearance",
    "has_free_space_path",
    "compute_nonconvexity_score_sdf",
]

