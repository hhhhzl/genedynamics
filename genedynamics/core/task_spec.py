"""
TaskSpec abstraction for dimension-agnostic position extraction and success criteria.

Phase 1: Abstraction only. Default implementation replicates legacy 2D behavior exactly
so existing results remain unchanged.

Usage:
    from genedynamics.core.task_spec import TaskSpec, get_default_task_spec

    spec = get_default_task_spec(env_plugin=env_plugin, env_name="single_integrator_box_2d")
    pos = spec.extract_position(state)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class TaskSpec(ABC):
    """
    Task-agnostic specification for position extraction and success criteria.

    Enables future 3D / high-dimensional / MJX / multi-robot extensions without
    changing solver logic. Phase 1: extract_position only; success_criterion
    added as optional hook for experiment layer.
    """

    @abstractmethod
    def extract_position(self, state: Any) -> np.ndarray:
        """
        Extract position coordinates from state.

        For 2D: returns (x, y). For 3D: (x, y, z). For high-dim: first N position dims.

        Args:
            state: Full state vector (array-like)

        Returns:
            Position vector, shape (pos_dim,) with pos_dim >= 1
        """
        pass

    @property
    def position_dim(self) -> int:
        """Expected position dimension (2 for 2D, 3 for 3D, etc.)."""
        return 2  # Legacy default

    def success_criterion(
        self,
        final_position: np.ndarray,
        target: Any,
        success_margin: float,
        *,
        env_name: Optional[str] = None,
    ) -> bool:
        """
        Check if task is successful (optional; experiment layer can override).

        Default: point target, ||final_pos - target|| < margin.
        d3il_avoiding_9d: line target, final_pos[1] >= target[1] - margin.

        Args:
            final_position: extract_position(final_state)
            target: Goal/target (array or object with extract_position)
            success_margin: Distance threshold
            env_name: Optional env identifier for special cases

        Returns:
            True if success
        """
        target_pos = np.asarray(target, dtype=np.float64).reshape(-1)
        pos_dim = min(3, max(2, len(np.asarray(final_position).reshape(-1))))
        if target_pos.size < pos_dim:
            target_pos = np.pad(target_pos, (0, pos_dim - target_pos.size), constant_values=0.0)
        final_pos = np.asarray(final_position, dtype=np.float64).reshape(-1)[:pos_dim]
        target_pos = target_pos[:pos_dim]
        env_lower = (env_name or "").lower()
        # D3IL avoiding (4D/9D): line target (y >= target_y - margin), 2D only
        if "d3il_avoiding" in env_lower and pos_dim >= 2:
            return bool(final_pos[1] >= target_pos[1] - success_margin)
        # 3D or 2D point target: ||final_pos - target|| < margin
        return bool(np.linalg.norm(final_pos - target_pos) < success_margin)


class Legacy3DTaskSpec(TaskSpec):
    """
    3D task spec for drone/quadrotor and other 3D environments.
    - State layout: [x, y, z, ...] (position in first 3 dims)
    - extract_position returns state[:3]
    """

    def extract_position(self, state: Any) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32)
        if len(state_np) >= 3:
            return state_np[:3].copy()
        return state_np[: min(3, len(state_np))].copy()

    @property
    def position_dim(self) -> int:
        return 3


class Legacy2DTaskSpec(TaskSpec):
    """
    Legacy 2D task spec. Replicates exact behavior of previous _default_extract_position:
    - len 4 (2D double integrator): state[:2]
    - len 2 (2D single integrator): state[:2]
    - else: state[:min(2, len)]
    """

    def extract_position(self, state: Any) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32)
        if len(state_np) == 4:  # 2D double integrator: [x, y, vx, vy]
            return state_np[:2].copy()
        if len(state_np) == 2:  # 2D single integrator: [x, y]
            return state_np[:2].copy()
        return state_np[: min(2, len(state_np))].copy()

    @property
    def position_dim(self) -> int:
        return 2


class EnvPluginTaskSpecAdapter(TaskSpec):
    """
    Wraps an EnvironmentPlugin to provide TaskSpec interface.
    Uses env_plugin.extract_position when available; else falls back to Legacy2DTaskSpec.
    """

    def __init__(self, env_plugin: Any, fallback: Optional[TaskSpec] = None):
        self.env_plugin = env_plugin
        self._fallback = fallback or Legacy2DTaskSpec()

    def extract_position(self, state: Any) -> np.ndarray:
        if self.env_plugin is not None and hasattr(self.env_plugin, "extract_position"):
            return np.asarray(self.env_plugin.extract_position(state), dtype=np.float32)
        return self._fallback.extract_position(state)

    def success_criterion(
        self,
        final_position: np.ndarray,
        target: Any,
        success_margin: float,
        *,
        env_name: Optional[str] = None,
    ) -> bool:
        env_name = env_name or getattr(self.env_plugin, "name", "")
        return super().success_criterion(
            final_position, target, success_margin, env_name=env_name
        )

    @property
    def position_dim(self) -> int:
        if self.env_plugin is not None and hasattr(self.env_plugin, "get_position_dim"):
            return int(self.env_plugin.get_position_dim())
        # Infer from env_plugin.name
        name = (getattr(self.env_plugin, "name", "") or "").lower()
        if "3d" in name or "drone" in name:
            return 3
        return 2


class SoftZooTaskSpec(TaskSpec):
    """
    TaskSpec for SoftZoo soft robot environments.

    Extracts position (centroid or control points) from high-dimensional
    deformable state. Default: first 2 dims for 2D planar, or first 3 for 3D.
    """

    def __init__(self, position_dim: int = 2, position_indices: Optional[tuple] = None):
        self._position_dim = int(position_dim)
        self._position_indices = position_indices  # None => use first N dims

    def extract_position(self, state: Any) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32).reshape(-1)
        if self._position_indices is not None:
            idx = np.asarray(self._position_indices, dtype=np.int32)
            idx = idx[idx >= 0]
            idx = idx[idx < len(state_np)]
            return state_np[idx].copy()
        return state_np[: min(self._position_dim, len(state_np))].copy()

    @property
    def position_dim(self) -> int:
        return self._position_dim


def get_default_task_spec(
    env_plugin: Optional[Any] = None,
    env_name: Optional[str] = None,
) -> TaskSpec:
    """
    Get default TaskSpec. Returns 2D/3D spec based on env_plugin and env_name.

    Args:
        env_plugin: If provided and has extract_position, wraps it
        env_name: Used to select 2D vs 3D when no env_plugin (e.g. "drone_full_3d" -> 3D)

    Returns:
        TaskSpec instance
    """
    env_name_lower = (env_name or "").lower()
    if "stepping_stones_2d" in env_name_lower and env_plugin is not None and hasattr(env_plugin, "extract_position"):
        return EnvPluginTaskSpecAdapter(env_plugin, fallback=Legacy2DTaskSpec())
    # SoftZoo: high-dim soft robot, use 2D centroid by default
    if "softzoo" in env_name_lower:
        return SoftZooTaskSpec(position_dim=2)
    # Quadruped: ant, go2 (flat [qpos; qvel], position = state[:3])
    if "quadruped" in env_name_lower:
        from genedynamics.tasks.quadruped.spec import QuadrupedTaskSpec
        return QuadrupedTaskSpec()
    # Humanoid: humanoid, g1 (flat [qpos; qvel], position = state[:3])
    if "humanoid" in env_name_lower:
        from genedynamics.tasks.humanoid.spec import HumanoidTaskSpec
        return HumanoidTaskSpec()
    # 3D env names: drone_full_3d, drone_box_3d, etc.
    is_3d = "drone_full_3d" in env_name_lower or "drone_box_3d" in env_name_lower or "3d" in env_name_lower
    if env_plugin is not None and hasattr(env_plugin, "extract_position"):
        fallback = Legacy3DTaskSpec() if is_3d else Legacy2DTaskSpec()
        return EnvPluginTaskSpecAdapter(env_plugin, fallback=fallback)
    return Legacy3DTaskSpec() if is_3d else Legacy2DTaskSpec()


def legacy_extract_position(state: Any) -> np.ndarray:
    """
    Standalone legacy position extractor. Same logic as Legacy2DTaskSpec.extract_position.
    Use when a callable is needed (e.g. CFS position_extractor).
    """
    return Legacy2DTaskSpec().extract_position(state)
