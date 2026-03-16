"""
Task configuration schema for deploy pipeline.

Industrial-grade task specification supporting:
- point target (legacy)
- obstacle avoidance (flat-ground)
- rough terrain (stairs/blocks + friction randomization)
- push recovery (randomized external impulses)
- velocity command (dial-mpc style)
- sequence target
- crate climb / crate push

All task types share this schema. Extensible via task_type and nested configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Nested task configs
# ---------------------------------------------------------------------------


@dataclass
class ObstacleConfig:
    """Obstacle configuration for obstacle_avoid task."""

    type: str = "sphere"  # "sphere" | "box" | "box3d"
    level: int = 0  # Difficulty level (0 = no obstacles)
    num_obstacles: int = 0  # Override level-based count when > 0
    robot_radius: float = 0.05
    obstacle_radius_scale: float = 1.0
    min_obstacle_margin: float = 0.1
    map_bounds: Optional[Dict[str, float]] = None  # x_min, x_max, y_min, y_max, z_min, z_max
    enable_connectivity_check: bool = True
    enable_nonconvexity_check: bool = True
    generator: Optional[str] = None  # "box2d" | "box3d" | "quadruped_3d"
    obstacle_radius_by_level: Optional[Dict[int, List[float]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObstacleConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            type=str(data.get("type", "sphere")),
            level=int(data.get("level", 0)),
            num_obstacles=int(data.get("num_obstacles", 0)),
            robot_radius=float(data.get("robot_radius", 0.05)),
            obstacle_radius_scale=float(data.get("obstacle_radius_scale", 1.0)),
            min_obstacle_margin=float(data.get("min_obstacle_margin", 0.1)),
            map_bounds=data.get("map_bounds"),
            enable_connectivity_check=bool(data.get("enable_connectivity_check", True)),
            enable_nonconvexity_check=bool(data.get("enable_nonconvexity_check", True)),
            generator=data.get("generator"),
            obstacle_radius_by_level=data.get("obstacle_radius_by_level"),
            extra={k: v for k, v in data.items() if k not in (
                "type", "level", "num_obstacles", "robot_radius", "obstacle_radius_scale",
                "min_obstacle_margin", "map_bounds", "enable_connectivity_check",
                "enable_nonconvexity_check", "generator", "obstacle_radius_by_level")},
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.type,
            "level": self.level,
            "num_obstacles": self.num_obstacles,
            "robot_radius": self.robot_radius,
            "obstacle_radius_scale": self.obstacle_radius_scale,
            "min_obstacle_margin": self.min_obstacle_margin,
            "enable_connectivity_check": self.enable_connectivity_check,
            "enable_nonconvexity_check": self.enable_nonconvexity_check,
            **self.extra,
        }
        if self.map_bounds is not None:
            out["map_bounds"] = self.map_bounds
        if self.generator is not None:
            out["generator"] = self.generator
        if self.obstacle_radius_by_level is not None:
            out["obstacle_radius_by_level"] = self.obstacle_radius_by_level
        return out


@dataclass
class TerrainConfig:
    """Terrain configuration for rough_terrain task."""

    type: str = "flat"  # "flat" | "stairs" | "blocks" | "hfield"
    friction_randomize: bool = False
    friction_range: Optional[List[float]] = None  # [min, max]
    stairs_step_height: float = 0.1
    stairs_step_depth: float = 0.3
    blocks_count: int = 5
    blocks_height_range: Optional[List[float]] = None  # [min, max]
    hfield_nrow: int = 64
    hfield_ncol: int = 64
    hfield_scale: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TerrainConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            type=str(data.get("type", "flat")),
            friction_randomize=bool(data.get("friction_randomize", False)),
            friction_range=data.get("friction_range"),
            stairs_step_height=float(data.get("stairs_step_height", 0.1)),
            stairs_step_depth=float(data.get("stairs_step_depth", 0.3)),
            blocks_count=int(data.get("blocks_count", 5)),
            blocks_height_range=data.get("blocks_height_range"),
            hfield_nrow=int(data.get("hfield_nrow", 64)),
            hfield_ncol=int(data.get("hfield_ncol", 64)),
            hfield_scale=float(data.get("hfield_scale", 1.0)),
            extra={k: v for k, v in data.items() if k not in (
                "type", "friction_randomize", "friction_range", "stairs_step_height",
                "stairs_step_depth", "blocks_count", "blocks_height_range",
                "hfield_nrow", "hfield_ncol", "hfield_scale")},
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.type,
            "friction_randomize": self.friction_randomize,
            "stairs_step_height": self.stairs_step_height,
            "stairs_step_depth": self.stairs_step_depth,
            "blocks_count": self.blocks_count,
            "hfield_nrow": self.hfield_nrow,
            "hfield_ncol": self.hfield_ncol,
            "hfield_scale": self.hfield_scale,
            **self.extra,
        }
        if self.friction_range is not None:
            out["friction_range"] = self.friction_range
        if self.blocks_height_range is not None:
            out["blocks_height_range"] = self.blocks_height_range
        return out


@dataclass
class PerturbationConfig:
    """Perturbation configuration for push_recovery task."""

    impulse_magnitude: float = 15.0
    interval_steps: Optional[List[int]] = None  # [min_step, max_step] for random impulse timing
    body: str = "torso"  # Body name or id to apply impulse
    direction_randomize: bool = True  # Random direction when True
    fixed_direction: Optional[List[float]] = None  # [x, y, z] when direction_randomize=False
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerturbationConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            impulse_magnitude=float(data.get("impulse_magnitude", 15.0)),
            interval_steps=data.get("interval_steps"),
            body=str(data.get("body", "torso")),
            direction_randomize=bool(data.get("direction_randomize", True)),
            fixed_direction=data.get("fixed_direction"),
            extra={k: v for k, v in data.items() if k not in (
                "impulse_magnitude", "interval_steps", "body",
                "direction_randomize", "fixed_direction")},
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "impulse_magnitude": self.impulse_magnitude,
            "body": self.body,
            "direction_randomize": self.direction_randomize,
            **self.extra,
        }
        if self.interval_steps is not None:
            out["interval_steps"] = self.interval_steps
        if self.fixed_direction is not None:
            out["fixed_direction"] = self.fixed_direction
        return out


@dataclass
class VelocityTaskConfig:
    """Velocity command task params (dial-mpc style)."""

    default_vx: float = 0.8
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 1.0
    gait: str = "trot"  # "trot" | "walk" | "jog" | "stand"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VelocityTaskConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            default_vx=float(data.get("default_vx", 0.8)),
            default_vy=float(data.get("default_vy", 0.0)),
            default_vyaw=float(data.get("default_vyaw", 0.0)),
            ramp_up_time=float(data.get("ramp_up_time", 1.0)),
            gait=str(data.get("gait", "trot")),
            extra={k: v for k, v in data.items() if k not in (
                "default_vx", "default_vy", "default_vyaw", "ramp_up_time", "gait")},
        )


@dataclass
class SequenceTaskConfig:
    """Sequence target task params."""

    pose_target_sequence: Optional[List[List[float]]] = None  # [[x,y,z], ...]
    yaw_target_sequence: Optional[List[float]] = None
    jump_dt: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SequenceTaskConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            pose_target_sequence=data.get("pose_target_sequence"),
            yaw_target_sequence=data.get("yaw_target_sequence"),
            jump_dt=float(data.get("jump_dt", 1.0)),
            extra={k: v for k, v in data.items() if k not in (
                "pose_target_sequence", "yaw_target_sequence", "jump_dt")},
        )


# ---------------------------------------------------------------------------
# Main TaskConfig
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """
    Unified task configuration.

    task_type: "point" | "obstacle_avoid" | "rough_terrain" | "push_recovery" |
               "velocity" | "sequence" | "crate_climb" | "crate_push"
    """

    task_type: str = "point"
    obstacles: Optional[ObstacleConfig] = None
    terrain: Optional[TerrainConfig] = None
    perturbation: Optional[PerturbationConfig] = None
    velocity: Optional[VelocityTaskConfig] = None
    sequence: Optional[SequenceTaskConfig] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        if not isinstance(data, dict):
            return cls()
        task_type = str(data.get("task_type", "point"))
        obstacles = None
        if "obstacles" in data:
            obstacles = ObstacleConfig.from_dict(data["obstacles"])
        elif "obstacle_config" in data:
            obstacles = ObstacleConfig.from_dict(data["obstacle_config"])
        terrain = None
        if "terrain" in data:
            terrain = TerrainConfig.from_dict(data["terrain"])
        perturbation = None
        if "perturbation" in data:
            perturbation = PerturbationConfig.from_dict(data["perturbation"])
        velocity = None
        if "velocity" in data or "task_params" in data:
            vdata = data.get("velocity") or data.get("task_params", {})
            velocity = VelocityTaskConfig.from_dict(vdata)
        sequence = None
        if "sequence" in data:
            sequence = SequenceTaskConfig.from_dict(data["sequence"])
        extra = {k: v for k, v in data.items() if k not in (
            "task_type", "obstacles", "obstacle_config", "terrain",
            "perturbation", "velocity", "task_params", "sequence")}
        return cls(
            task_type=task_type,
            obstacles=obstacles,
            terrain=terrain,
            perturbation=perturbation,
            velocity=velocity,
            sequence=sequence,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "task_type": self.task_type,
            **self.extra,
        }
        if self.obstacles is not None:
            out["obstacles"] = self.obstacles.to_dict()
        if self.terrain is not None:
            out["terrain"] = self.terrain.to_dict()
        if self.perturbation is not None:
            out["perturbation"] = self.perturbation.to_dict()
        if self.velocity is not None:
            out["velocity"] = {
                "default_vx": self.velocity.default_vx,
                "default_vy": self.velocity.default_vy,
                "default_vyaw": self.velocity.default_vyaw,
                "ramp_up_time": self.velocity.ramp_up_time,
                "gait": self.velocity.gait,
                **self.velocity.extra,
            }
        if self.sequence is not None:
            out["sequence"] = {
                "pose_target_sequence": self.sequence.pose_target_sequence,
                "yaw_target_sequence": self.sequence.yaw_target_sequence,
                "jump_dt": self.sequence.jump_dt,
                **self.sequence.extra,
            }
        return out
