"""
Quadruped robot profiles (ant, flat, rough, push, go2).

Supports task-driven env/planner: obstacle_avoid, rough_terrain, push_recovery.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from genedynamics.deploy.profiles.base import RobotProfile, ProfileRegistry
from genedynamics.robots import get_robot_registry


def _make_quadruped_env(config: Dict[str, Any], **overrides: Any) -> Any:
    from genedynamics.deploy.factory import make_quadruped_env
    from genedynamics.deploy.obstacles import make_obstacles_from_task

    params = dict(config.get("env_params", {}))
    params.update(overrides)
    model = params.pop("model", config.get("model_id", "flat"))
    use_mjx = params.pop("use_mjx", False)
    env_name = params.pop("env_name", None)
    use_brax = "brax" in (env_name or "").lower() or params.get("physics_backend") == "brax"
    if use_brax and not env_name:
        env_name = "quadruped_go2_brax" if model == "go2" else "humanoid_run_brax"
    task = config.get("task")
    target = tuple(params.get("target", (2.0, 0.0, 0.5)))

    # Task-driven model/env selection (Phase 3: rough_terrain, push_recovery)
    if task is not None:
        task_type = getattr(task, "task_type", "point")
        if task_type == "rough_terrain" and model in ("ant", "flat"):
            model = "rough"
        elif task_type == "push_recovery" and model in ("ant", "flat"):
            model = "push"

    # Obstacle_avoid: create obstacles before env (for envs that accept them at init)
    obstacles = None
    if task is not None and getattr(task, "task_type", "") == "obstacle_avoid" and getattr(task, "obstacles", None):
        obstacles = make_obstacles_from_task(
            task,
            seed=config.get("seed", 0),
            start_pos=(0.0, 0.0, 0.5),
            target_pos=target,
            robot_radius_override=params.get("robot_radius"),
        )
        if obstacles is not None:
            params["obstacles"] = obstacles

    env = make_quadruped_env(
        model=model,
        env_name=env_name,
        use_mjx=use_mjx or use_brax,
        **params,
    )

    # Fallback: inject obstacles if env did not accept at init
    if obstacles is not None and hasattr(env, "obstacles") and getattr(env, "obstacles", None) is None:
        env.obstacles = obstacles

    # Terrain/perturbation params (for physics backends that support them)
    if task is not None and hasattr(env, "terrain_type"):
        if getattr(task, "terrain", None):
            env.terrain_type = getattr(task.terrain, "type", "flat")
    if task is not None and hasattr(env, "task_type"):
        if getattr(task, "perturbation", None):
            env.task_type = "push"

    # Push recovery: wrap env with perturbation applicator
    if task is not None and getattr(task, "perturbation", None):
        from genedynamics.deploy.perturbation import wrap_env_with_perturbation
        env = wrap_env_with_perturbation(env, task.perturbation)

    return env


def _make_quadruped_energy(config: Dict[str, Any], env: Any = None) -> Any:
    from genedynamics.deploy.factory import make_quadruped_energy

    params = config.get("env_params", {})
    target = tuple(params.get("target", (2.0, 0.0, 0.5)))
    env_name = "quadruped_flat_mjx" if (env and "mjx" in str(type(env).__name__).lower()) else "quadruped_flat_physics"
    if env and "go2" in str(getattr(env, "model", "")).lower():
        env_name = "quadruped_go2_mjx" if "mjx" in str(type(env).__name__).lower() else "quadruped_go2_physics"
    if env and "brax" in str(type(env).__name__).lower():
        env_name = "quadruped_go2_brax"
    return make_quadruped_energy(target=target, env_name=env_name)


def _make_quadruped_planner(env: Any, config: Dict[str, Any], **overrides: Any) -> Any:
    from genedynamics.deploy.factory import make_quadruped_planner
    from genedynamics.deploy.obstacles import make_obstacles_from_task

    planner = config.get("planner", "stand")
    horizon = config.get("horizon", 64)
    use_mjx = "mjx" in str(type(env).__name__).lower() or "brax" in str(type(env).__name__).lower()
    params = dict(config.get("method_params", {}))
    params.update(overrides)
    task = config.get("task")

    obstacles: Optional[Any] = None
    if task is not None and getattr(task, "task_type", "") == "obstacle_avoid":
        obstacles = getattr(env, "obstacles", None)
        if obstacles is None and hasattr(env, "_base"):
            obstacles = getattr(env._base, "obstacles", None)
        if obstacles is None:
            obstacles = make_obstacles_from_task(
                task,
                seed=config.get("seed", 0),
                start_pos=config.get("env_params", {}).get("target"),
                target_pos=config.get("env_params", {}).get("target", (2.0, 0.0, 0.5)),
            )

    # Create scheduler from scheduler_config (aligns with single_2d)
    scheduler = None
    scheduler_config = config.get("scheduler_config")
    if scheduler_config:
        from genedynamics.experiments.common.constraints import create_scheduler_from_config
        import os
        backend = "jax" if os.environ.get("GENEDYNAMICS_BACKEND", "jax") == "jax" else "numpy"
        obstacle_config = {"robot_radius": 0.15}
        if task is not None:
            obs = getattr(task, "obstacles", None)
            if obs is not None and hasattr(obs, "robot_radius"):
                obstacle_config["robot_radius"] = obs.robot_radius
            elif isinstance(getattr(task, "obstacles", None), dict):
                obstacle_config["robot_radius"] = task.obstacles.get("robot_radius", 0.15)
        scheduler = create_scheduler_from_config(
            scheduler_config,
            backend,
            method_params=params,
            obstacle_config=obstacle_config,
        )
        params["scheduler"] = scheduler

    return make_quadruped_planner(
        env=env,
        horizon=horizon,
        planner_type=planner,
        use_mjx=use_mjx,
        obstacles=obstacles,
        **params,
    )


def _infer_quadruped_spec(env: Any) -> Tuple[int, int, int]:
    nq = getattr(env, "nq", 15)
    nv = getattr(env, "nv", 14)
    act_dim = getattr(env, "act_dim", 8)
    return nq, nv, act_dim


def create_quadruped_profiles(reg: ProfileRegistry) -> None:
    """Register quadruped profiles."""
    robot_reg = get_robot_registry()
    for model_id in ("ant", "flat", "rough", "push", "go2"):
        entry = robot_reg.get("quadruped", model_id)
        if entry is None:
            continue
        profile = RobotProfile(
            robot_type="quadruped",
            model_id=model_id,
            entry=entry,
            make_env=_make_quadruped_env,
            make_energy=_make_quadruped_energy,
            make_planner=_make_quadruped_planner,
            infer_spec=_infer_quadruped_spec,
            planners_requiring_mjx=("mbd", "2go", "cfsmbd", "cfsmbd_full", "mdoc"),
            default_planner="stand",
        )
        reg.register(profile)
