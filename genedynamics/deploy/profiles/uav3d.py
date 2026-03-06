"""
UAV 3D robot profile.

UAV 3D is a special case: single "model" (drone), planners 2go/mbd.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from genedynamics.deploy.profiles.base import RobotProfile, ProfileRegistry


def _make_uav3d_env(config: Dict[str, Any], **overrides: Any) -> Any:
    from genedynamics.deploy.factory import make_uav3d_mjx_env

    params = dict(config.get("env_params", {}))
    params.update(overrides)
    # Filter out quadruped/legged-specific keys that UAV env doesn't accept
    for k in ("env_name", "model", "use_mjx"):
        params.pop(k, None)
    return make_uav3d_mjx_env(**params)


def _make_uav3d_energy(config: Dict[str, Any], env: Any = None) -> Any:
    from genedynamics.deploy.factory import make_uav3d_energy

    params = config.get("env_params", {})
    target = tuple(params.get("target", (0.0, 0.0, 1.0)))
    return make_uav3d_energy(target=target)


def _make_uav3d_planner(env: Any, config: Dict[str, Any], **overrides: Any) -> Any:
    from genedynamics.deploy.factory import make_2go_planner, make_mbd_planner, make_uav3d_energy

    planner = config.get("planner", "2go")
    horizon = config.get("horizon", 64)
    target = tuple(config.get("env_params", {}).get("target", (0.0, 0.0, 1.0)))
    energy = make_uav3d_energy(target=target)
    params = dict(config.get("method_params", {}))
    params.update(overrides)

    if planner.lower() == "mbd":
        return make_mbd_planner(
            env=env,
            energy=energy,
            horizon=horizon,
            dt=config.get("env_params", {}).get("dt", 0.05),
            seed=config.get("seed", 0),
            env_name="drone_full_3d_physics",
            **params,
        )
    return make_2go_planner(
        env=env,
        energy=energy,
        obstacles=None,
        horizon=horizon,
        dt=config.get("env_params", {}).get("dt", 0.05),
        seed=config.get("seed", 0),
        **params,
    )


def _infer_uav3d_spec(env: Any) -> Tuple[int, int, int]:
    # UAV 3D: 6 pos + 6 vel = 12, act_dim 4
    nq = getattr(env, "nq", 6)
    nv = getattr(env, "nv", 6)
    act_dim = getattr(env, "act_dim", 4)
    return nq, nv, act_dim


def create_uav3d_profiles(reg: ProfileRegistry) -> None:
    """Register UAV 3D profile."""
    # UAV 3D has no RobotEntry; we create a synthetic entry
    from genedynamics.robots.registry import RobotEntry

    entry = RobotEntry(
        robot_type="uav3d",
        model_id="drone",
        env_factory_name="drone_full_3d_mjx",
        nq=6,
        nv=6,
        act_dim=4,
        model_path_resolver=None,
        spec_class=None,
        description="UAV 3D (drone)",
    )
    profile = RobotProfile(
        robot_type="uav3d",
        model_id="drone",
        entry=entry,
        make_env=_make_uav3d_env,
        make_energy=_make_uav3d_energy,
        make_planner=_make_uav3d_planner,
        infer_spec=_infer_uav3d_spec,
        planners_requiring_mjx=("mbd", "2go"),
        default_planner="2go",
    )
    reg.register(profile)
