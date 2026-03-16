"""
Humanoid robot profiles (humanoid, h1, g1).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from genedynamics.deploy.profiles.base import RobotProfile, ProfileRegistry
from genedynamics.robots import get_robot_registry


def _make_humanoid_env(config: Dict[str, Any], **overrides: Any) -> Any:
    from genedynamics.deploy.factory import make_humanoid_env

    params = dict(config.get("env_params", {}))
    params.update(overrides)
    model = params.pop("model", config.get("model_id", "humanoid"))
    use_mjx = params.pop("use_mjx", False)
    env_name = params.pop("env_name", None)
    return make_humanoid_env(
        model=model,
        env_name=env_name,
        use_mjx=use_mjx,
        **params,
    )


def _make_humanoid_energy(config: Dict[str, Any], env: Any = None) -> Any:
    from genedynamics.deploy.factory import make_humanoid_energy

    params = config.get("env_params", {})
    target = tuple(params.get("target", (3.0, 0.0, 1.0)))
    env_name = "humanoid_simplified_mjx" if (env and "mjx" in str(type(env).__name__).lower()) else "humanoid_simplified_physics"
    return make_humanoid_energy(target=target, env_name=env_name)


def _make_humanoid_planner(env: Any, config: Dict[str, Any], **overrides: Any) -> Any:
    from genedynamics.deploy.factory import make_humanoid_planner

    planner = config.get("planner", "stand")
    horizon = config.get("horizon", 64)
    use_mjx = "mjx" in str(type(env).__name__).lower()
    params = dict(config.get("method_params", {}))
    params.update(overrides)
    return make_humanoid_planner(
        env=env,
        horizon=horizon,
        planner_type=planner,
        use_mjx=use_mjx,
        **params,
    )


def _infer_humanoid_spec(env: Any) -> Tuple[int, int, int]:
    nq = getattr(env, "nq", 24)
    nv = getattr(env, "nv", 23)
    act_dim = getattr(env, "act_dim", 17)
    return nq, nv, act_dim


def create_humanoid_profiles(reg: ProfileRegistry) -> None:
    """Register humanoid profiles."""
    robot_reg = get_robot_registry()
    for model_id in ("humanoid", "h1", "g1"):
        entry = robot_reg.get("humanoid", model_id)
        if entry is None:
            continue
        profile = RobotProfile(
            robot_type="humanoid",
            model_id=model_id,
            entry=entry,
            make_env=_make_humanoid_env,
            make_energy=_make_humanoid_energy,
            make_planner=_make_humanoid_planner,
            infer_spec=_infer_humanoid_spec,
            planners_requiring_mjx=("mbd", "2go", "cfsmbd", "cfsmbd_full", "mdoc"),
            default_planner="stand",
        )
        reg.register(profile)
