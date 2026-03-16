"""
Base RobotProfile and registry.

RobotProfile extends RobotEntry with factory methods for env, energy, planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from genedynamics.robots.registry import RobotEntry, get_robot_registry


_profile_registry: Optional["ProfileRegistry"] = None


@dataclass
class RobotProfile:
    """
    Robot profile for deploy pipeline.

    Extends RobotEntry with factory methods. Each (robot_type, model_id)
    has one profile that knows how to build env, energy, planner.
    """

    robot_type: str
    model_id: str
    entry: RobotEntry

    # Factory methods: (config: Dict) -> Any
    make_env: Callable[..., Any]
    make_energy: Callable[..., Any]
    make_planner: Callable[..., Any]

    # Infer (nq, nv, act_dim) from env
    infer_spec: Callable[[Any], Tuple[int, int, int]] = field(default_factory=lambda: _default_infer_spec)

    # Optional: planners that require MJX (use_mjx=True for env)
    planners_requiring_mjx: Tuple[str, ...] = ("mbd", "2go", "cfsmbd", "mdoc")

    # Optional: default planner when not specified
    default_planner: str = "stand"

    # Optional: default env_name override
    default_env_name: Optional[str] = None

    def get_nq_nv_act_dim(self) -> Tuple[int, int, int]:
        """Get from entry."""
        return self.entry.nq, self.entry.nv, self.entry.act_dim

    def requires_mjx(self, planner: str) -> bool:
        """Whether this planner needs MJX env."""
        return planner.lower() in (p.lower() for p in self.planners_requiring_mjx)


def _default_infer_spec(env: Any) -> Tuple[int, int, int]:
    """Default spec inference from env."""
    nq = getattr(env, "nq", None)
    nv = getattr(env, "nv", None)
    act_dim = getattr(env, "act_dim", 4)
    if nq is not None and nv is not None:
        return nq, nv, act_dim
    state_dim = getattr(env, "state_dim", None)
    if state_dim is not None:
        nq = state_dim // 2
        nv = state_dim - nq
        return nq, nv, act_dim
    return 15, 14, 8  # Quadruped default fallback


class ProfileRegistry:
    """
    Registry for RobotProfiles.

    Maps (robot_type, model_id) -> RobotProfile.
    """

    def __init__(self) -> None:
        self._profiles: Dict[Tuple[str, str], RobotProfile] = {}

    def register(self, profile: RobotProfile) -> None:
        """Register a profile."""
        key = (profile.robot_type, profile.model_id)
        self._profiles[key] = profile

    def get(self, robot_type: str, model_id: str) -> Optional[RobotProfile]:
        """Get profile by robot_type and model_id."""
        return self._profiles.get((robot_type, model_id))

    def list_robot_types(self) -> List[str]:
        """List registered robot types."""
        return list({rt for rt, _ in self._profiles.keys()})

    def list_models(self, robot_type: str) -> List[str]:
        """List model IDs for a robot type."""
        return [mid for (rt, mid) in self._profiles.keys() if rt == robot_type]

    def require(self, robot_type: str, model_id: str) -> RobotProfile:
        """Get profile or raise."""
        p = self.get(robot_type, model_id)
        if p is None:
            available = self.list_models(robot_type)
            raise ValueError(
                f"No profile for ({robot_type}, {model_id}). "
                f"Available models: {available}"
            )
        return p


def get_profile_registry() -> ProfileRegistry:
    """Get singleton profile registry."""
    global _profile_registry
    if _profile_registry is None:
        _profile_registry = ProfileRegistry()
    return _profile_registry
