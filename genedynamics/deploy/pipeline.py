"""
Deploy pipeline: orchestrates config, profile, mode, and execution.

Single entry point for all deploy operations. Loads config, resolves profile,
builds env and planner, dispatches to mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from genedynamics.deploy.config import DeployConfig
from genedynamics.deploy.profiles import get_profile_registry
from genedynamics.deploy.modes import get_mode


def run_pipeline(
    config: Union[DeployConfig, str, Path, Dict[str, Any]],
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run deploy pipeline.

    Args:
        config: DeployConfig, path to YAML, or dict
        overrides: Optional dict to override config values

    Returns:
        Result dict from mode.run()
    """
    cfg = _resolve_config(config, overrides)
    _setup_environment(cfg)

    profile_reg = get_profile_registry()
    profile = profile_reg.require(cfg.robot_type, cfg.model_id)

    # Build env with use_mjx when planner requires it
    use_mjx = profile.requires_mjx(cfg.planner)
    config_dict = _config_to_dict(cfg)
    config_dict["env_params"] = dict(cfg.env_params)
    config_dict["env_params"]["model"] = cfg.model_id
    config_dict["env_params"]["use_mjx"] = use_mjx

    env = profile.make_env(config_dict)

    # Build planner
    planner = profile.make_planner(env, config_dict)

    # Dispatch to mode
    mode = get_mode(cfg.mode)
    result = mode.run(cfg, profile, env, planner)

    result["config"] = cfg.to_dict()
    result["robot_type"] = cfg.robot_type
    result["model_id"] = cfg.model_id
    result["output_dir"] = cfg.output_dir

    return result


def _resolve_config(
    config: Union[DeployConfig, str, Path, Dict[str, Any]],
    overrides: Optional[Dict[str, Any]] = None,
) -> DeployConfig:
    """Resolve config from various inputs."""
    if isinstance(config, DeployConfig):
        cfg = config
    elif isinstance(config, (str, Path)):
        cfg = DeployConfig.from_yaml(config)
    elif isinstance(config, dict):
        cfg = DeployConfig.from_dict(config)
    else:
        raise TypeError(f"config must be DeployConfig, path, or dict, got {type(config)}")

    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    return cfg


def _apply_overrides(cfg: DeployConfig, overrides: Dict[str, Any]) -> DeployConfig:
    """Apply overrides to config."""
    data = cfg.to_dict()
    for k, v in overrides.items():
        if k == "env_params":
            data["env_params"] = {**data.get("env_params", {}), **v}
        elif k == "method_params":
            data["method_params"] = {**data.get("method_params", {}), **v}
        elif k == "start" and isinstance(v, dict):
            data["start"] = {**data.get("start", {}), **v}
        elif k == "replay" and isinstance(v, dict):
            data["replay"] = {**data.get("replay", {}), **v}
        elif k in data:
            data[k] = v
        else:
            data.setdefault("extra", {})[k] = v
    return DeployConfig.from_dict(data)


def _config_to_dict(cfg: DeployConfig) -> Dict[str, Any]:
    """Convert config to dict for profile factories."""
    return {
        "robot_type": cfg.robot_type,
        "model_id": cfg.model_id,
        "mode": cfg.mode,
        "planner": cfg.planner,
        "horizon": cfg.horizon,
        "plan_mode": cfg.plan_mode,
        "control_rate_hz": cfg.control_rate_hz,
        "plan_rate_hz": cfg.plan_rate_hz,
        "max_steps": cfg.max_steps,
        "episodes": cfg.episodes,
        "seed": cfg.seed,
        "record": cfg.record,
        "output_dir": cfg.output_dir,
        "tags": cfg.tags,
        "env_params": cfg.env_params,
        "method_params": cfg.method_params,
        "start": cfg.start,
        "sim": cfg.sim,
        "real": cfg.real,
        "replay": cfg.replay,
        "task": cfg.task,
        "extra": cfg.extra,
    }


def _setup_environment(cfg: DeployConfig) -> None:
    """Set up process environment (MUJOCO_GL, etc.)."""
    if "MUJOCO_GL" not in os.environ:
        import platform
        if platform.system() == "Linux":
            os.environ.setdefault("MUJOCO_GL", "egl")
        elif platform.system() == "Darwin":
            os.environ.setdefault("MUJOCO_GL", "glfw")

    # Ensure output dir exists
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if cfg.record and cfg.get_episode_dir():
        Path(cfg.get_episode_dir()).mkdir(parents=True, exist_ok=True)
