"""
Unified deploy configuration schema.

Industrial-grade config with validation, defaults, and mode-specific extensions.
All deploy modes (sim, real, shadow, replay) share this schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from genedynamics.deploy.task_config import TaskConfig


# ---------------------------------------------------------------------------
# Mode-specific configs
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Simulation-specific configuration."""

    real_time_factor: float = 1.0
    sync_mode: bool = True
    sim_dt: float = 0.0  # 0 = infer from control_rate_hz
    headless: bool = False
    viewer: bool = False  # Launch MuJoCo viewer during sim


@dataclass
class RealConfig:
    """Real robot deployment configuration."""

    localization_plugin: str = "mock"
    localization_timeout_sec: float = 2.0
    network_interface: str = ""
    low_cmd_pub_dt: float = 0.002
    initial_position_ctrl: Optional[List[float]] = None
    real_kp: Union[float, List[float]] = 40.0
    real_kd: Union[float, List[float]] = 1.0


@dataclass
class ReplayConfig:
    """Replay mode configuration."""

    episode_dir: str = ""
    episode_pattern: str = "ep_*"  # Glob for multiple episodes
    real_time_factor: float = 1.0
    render: bool = True
    output_format: str = "gif"  # "gif" | "html" | "none"


# ---------------------------------------------------------------------------
# Start mode for initial state
# ---------------------------------------------------------------------------


@dataclass
class StartConfig:
    """Initial state / spawn configuration."""

    mode: str = "random"  # "random" | "near_target" | "fixed"
    near_target_min_dist: float = 0.5
    near_target_max_dist: float = 1.0
    fixed_state: Optional[List[float]] = None
    start_min_z: float = 0.4  # For legged robots


# ---------------------------------------------------------------------------
# Main DeployConfig
# ---------------------------------------------------------------------------


@dataclass
class DeployConfig:
    """
    Unified deploy configuration.

    Single schema for all robot types and execution modes.
    Load from YAML or construct programmatically.
    """

    # Identity
    name: str = "deploy"
    robot_type: str = "quadruped"  # "quadruped" | "humanoid" | "uav3d"
    model_id: str = "flat"  # "ant", "go2", "g1", "humanoid", etc.
    mode: str = "sim"  # "sim" | "real" | "shadow" | "replay"
    planner: str = "stand"  # "stand" | "mbd" | "2go" | "cfsmbd" | ...

    # Planning
    horizon: int = 64
    plan_mode: str = "plan_once"  # "plan_once" | "mpc"

    # Execution
    control_rate_hz: float = 20.0
    plan_rate_hz: float = 10.0
    max_steps: int = 150
    episodes: int = 1
    seed: int = 0

    # Recording
    record: bool = True
    output_dir: str = "results/deploy"
    tags: Dict[str, str] = field(default_factory=dict)

    # Environment params (passed to make_env)
    env_params: Dict[str, Any] = field(default_factory=dict)

    # Method/planner params (Nsample, Ndiffuse, etc.)
    method_params: Dict[str, Any] = field(default_factory=dict)

    # Mode-specific
    sim: Optional[SimConfig] = None
    real: Optional[RealConfig] = None
    replay: Optional[ReplayConfig] = None

    # Start config
    start: Optional[StartConfig] = None

    # Task config (obstacles, terrain, perturbation, velocity, sequence)
    task: Optional[TaskConfig] = None

    # Scheduler config (diffusion_schedulers, constraint_schedulers) - aligns with single_2d
    scheduler_config: Optional[Dict[str, Any]] = None

    # Extra (passthrough for backward compat)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure mode-specific configs exist when needed."""
        if self.mode == "sim" and self.sim is None:
            self.sim = SimConfig()
        if self.mode == "real" and self.real is None:
            self.real = RealConfig()
        if self.mode == "replay" and self.replay is None:
            self.replay = ReplayConfig()
        if self.start is None:
            self.start = StartConfig()

    def get_episode_dir(self) -> Optional[str]:
        """Episode directory when recording."""
        if not self.record:
            return None
        return str(Path(self.output_dir) / "episodes")

    def get_session_config_extra(self) -> Dict[str, Any]:
        """Extra dict for SessionConfig (action_smooth_alpha, etc.)."""
        out = dict(self.extra)
        out.update(self.method_params)
        return out

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DeployConfig":
        """Load config from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeployConfig":
        """Build DeployConfig from dict (e.g. from YAML)."""
        # Extract top-level fields
        kwargs: Dict[str, Any] = {}
        for key in (
            "name", "robot_type", "model_id", "mode", "planner",
            "horizon", "plan_mode", "control_rate_hz", "plan_rate_hz",
            "max_steps", "episodes", "seed", "record", "output_dir", "tags",
        ):
            if key in data:
                kwargs[key] = data[key]

        # Legacy: execution_mode -> mode
        if "mode" not in kwargs and "execution_mode" in data:
            em = data["execution_mode"]
            if em in ("sim_only", "sim"):
                kwargs["mode"] = "sim"
            elif em == "shadow":
                kwargs["mode"] = "shadow"
            elif em in ("real_only", "real"):
                kwargs["mode"] = "real"
            elif em == "replay":
                kwargs["mode"] = "replay"
            else:
                kwargs["mode"] = str(em).lower() if em else "sim"

        # Backward compat: infer robot_type/model_id from env_name
        if "robot_type" not in kwargs:
            env_name = str(data.get("env_name", ""))
            if "drone" in env_name or "uav" in env_name.lower():
                kwargs["robot_type"] = "uav3d"
                kwargs["model_id"] = "drone"
            elif "quadruped" in env_name or "go2" in env_name or "ant" in env_name:
                kwargs["robot_type"] = "quadruped"
                kwargs["model_id"] = "go2" if "go2" in env_name else "flat"
            elif "humanoid" in env_name or "g1" in env_name or "h1" in env_name:
                kwargs["robot_type"] = "humanoid"
                kwargs["model_id"] = "g1" if "g1" in env_name else ("h1" if "h1" in env_name else "humanoid")
        if "planner" not in kwargs and "method" in data:
            kwargs["planner"] = data["method"]

        # env_params
        kwargs["env_params"] = dict(data.get("env_params", {}))
        # Merge top-level env keys into env_params for backward compat
        for k in ("dt", "target", "control_limit", "model", "env_name", "physics_backend"):
            if k in data and k not in kwargs["env_params"]:
                kwargs["env_params"][k] = data[k]

        # method_params
        kwargs["method_params"] = dict(data.get("method_params", {}))
        for k in ("Nsample", "Ndiffuse", "temp_sample", "terminal_energy_weight",
                  "show_tqdm", "action_smooth_alpha"):
            if k in data and k not in kwargs["method_params"]:
                kwargs["method_params"][k] = data[k]

        # start
        start_data = data.get("start", {})
        if isinstance(start_data, dict):
            kwargs["start"] = StartConfig(
                mode=start_data.get("mode", "random"),
                near_target_min_dist=float(start_data.get("near_target_min_dist", 0.5)),
                near_target_max_dist=float(start_data.get("near_target_max_dist", 1.0)),
                start_min_z=float(start_data.get("start_min_z", 0.4)),
            )
        # Legacy: start_mode, start_min_dist, start_max_dist
        if "start_mode" in data:
            kwargs["start"].mode = data["start_mode"]
        if "start_min_dist" in data:
            kwargs["start"].near_target_min_dist = float(data["start_min_dist"])
        if "start_max_dist" in data:
            kwargs["start"].near_target_max_dist = float(data["start_max_dist"])

        # sim
        sim_data = data.get("sim", {})
        if isinstance(sim_data, dict):
            kwargs["sim"] = SimConfig(
                real_time_factor=float(sim_data.get("real_time_factor", 1.0)),
                sync_mode=bool(sim_data.get("sync_mode", True)),
                sim_dt=float(sim_data.get("sim_dt", 0.0)),
                headless=bool(sim_data.get("headless", False)),
                viewer=bool(sim_data.get("viewer", False)),
            )
        # Legacy: real_time_factor at top level
        if "real_time_factor" in data and kwargs.get("sim"):
            kwargs["sim"].real_time_factor = float(data["real_time_factor"])

        # real
        real_data = data.get("real", {})
        if isinstance(real_data, dict):
            kwargs["real"] = RealConfig(
                localization_plugin=str(real_data.get("localization_plugin", "mock")),
                localization_timeout_sec=float(real_data.get("localization_timeout_sec", 2.0)),
                network_interface=str(real_data.get("network_interface", "")),
                low_cmd_pub_dt=float(real_data.get("low_cmd_pub_dt", 0.002)),
                initial_position_ctrl=real_data.get("initial_position_ctrl"),
                real_kp=real_data.get("real_kp", 40.0),
                real_kd=real_data.get("real_kd", 1.0),
            )

        # replay
        replay_data = data.get("replay", {})
        if isinstance(replay_data, dict):
            kwargs["replay"] = ReplayConfig(
                episode_dir=str(replay_data.get("episode_dir", "")),
                episode_pattern=str(replay_data.get("episode_pattern", "ep_*")),
                real_time_factor=float(replay_data.get("real_time_factor", 1.0)),
                render=bool(replay_data.get("render", True)),
                output_format=str(replay_data.get("output_format", "gif")),
            )

        # task (obstacles, terrain, perturbation, velocity, sequence)
        task_data = data.get("task", data.get("task_config", {}))
        if isinstance(task_data, dict) and task_data:
            kwargs["task"] = TaskConfig.from_dict(task_data)
        elif "obstacles" in data or "terrain" in data or "perturbation" in data:
            kwargs["task"] = TaskConfig.from_dict({
                "task_type": data.get("task_type", "point"),
                "obstacles": data.get("obstacles"),
                "terrain": data.get("terrain"),
                "perturbation": data.get("perturbation"),
                "velocity": data.get("velocity", data.get("task_params")),
                "sequence": data.get("sequence"),
            })

        # scheduler_config (aligns with experiment/single_2d)
        if "scheduler_config" in data:
            kwargs["scheduler_config"] = data["scheduler_config"]

        kwargs["extra"] = {k: v for k, v in data.items()
                          if k not in kwargs and k not in (
                              "env_params", "method_params", "start", "sim", "real", "replay",
                              "task", "task_config", "obstacles", "terrain", "perturbation",
                              "velocity", "sequence", "task_type", "task_params", "scheduler_config")}

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for YAML/JSON."""
        out: Dict[str, Any] = {
            "name": self.name,
            "robot_type": self.robot_type,
            "model_id": self.model_id,
            "mode": self.mode,
            "planner": self.planner,
            "horizon": self.horizon,
            "plan_mode": self.plan_mode,
            "control_rate_hz": self.control_rate_hz,
            "plan_rate_hz": self.plan_rate_hz,
            "max_steps": self.max_steps,
            "episodes": self.episodes,
            "seed": self.seed,
            "record": self.record,
            "output_dir": self.output_dir,
            "tags": self.tags,
            "env_params": self.env_params,
            "method_params": self.method_params,
            **self.extra,
        }
        if self.sim:
            out["sim"] = {
                "real_time_factor": self.sim.real_time_factor,
                "sync_mode": self.sim.sync_mode,
                "sim_dt": self.sim.sim_dt,
                "headless": self.sim.headless,
                "viewer": self.sim.viewer,
            }
        if self.real:
            out["real"] = {
                "localization_plugin": self.real.localization_plugin,
                "localization_timeout_sec": self.real.localization_timeout_sec,
                "network_interface": self.real.network_interface,
            }
        if self.replay:
            out["replay"] = {
                "episode_dir": self.replay.episode_dir,
                "episode_pattern": self.replay.episode_pattern,
                "real_time_factor": self.replay.real_time_factor,
                "render": self.replay.render,
                "output_format": self.replay.output_format,
            }
        if self.start:
            out["start"] = {
                "mode": self.start.mode,
                "near_target_min_dist": self.start.near_target_min_dist,
                "near_target_max_dist": self.start.near_target_max_dist,
                "start_min_z": self.start.start_min_z,
            }
        if self.task is not None:
            out["task"] = self.task.to_dict()
        if hasattr(self, "scheduler_config") and self.scheduler_config is not None:
            out["scheduler_config"] = self.scheduler_config
        return out
