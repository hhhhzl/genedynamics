"""
Factory for building execution pipeline: env, planner, executor.

Creates UAV 3D MJX env, 2GO planner, and executor with all components.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Headless for CI
if "MUJOCO_GL" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")


def _mjx_available() -> bool:
    """Check if MJX (mujoco-mjx) is available. False on x86_64 Python on Apple Silicon."""
    try:
        import mujoco
        from mujoco import mjx  # noqa: F401
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


def make_uav3d_mjx_env(
    dt: float = 0.05,
    horizon: int = 64,
    target: tuple = (0.0, 0.0, 1.0),
    control_limit: float = 1.0,
    obstacles: Optional[Any] = None,
    force_mjx: bool = False,
    force_drone_model: bool = False,
    **kwargs: Any,
) -> Any:
    """Create UAV 3D environment. Uses MJX when available, else drone_model (NumPy)."""
    from genedynamics.envs.factories import make_env

    use_mjx = (not force_drone_model) and (_mjx_available() or force_mjx)
    env_name = "drone_full_3d_mjx" if use_mjx else "drone_full_3d_physics"
    config = {
        "dt": dt,
        "horizon": horizon,
        "target": target,
        "control_limit": control_limit,
        "physics_backend": "mjx" if use_mjx else "drone_model",
        **kwargs,
    }
    if obstacles is not None:
        config["obstacles"] = obstacles
    if not use_mjx:
        import warnings
        warnings.warn(
            "MJX unavailable (e.g. x86_64 Python on Apple Silicon). "
            "Using drone_model (NumPy). For MJX, use native arm64 Python.",
            UserWarning,
        )
    return make_env(env_name, **config)


def make_uav3d_energy(target: tuple = (0.0, 0.0, 1.0)) -> Any:
    """Create energy functional for UAV 3D."""
    from genedynamics.envs.factories import make_energy

    return make_energy("drone_full_3d_physics")


def make_quadruped_env(
    env_name: Optional[str] = None,
    model: Optional[str] = None,
    dt: float = 0.05,
    horizon: int = 200,
    target: tuple = (2.0, 0.0, 0.5),
    control_limit: float = 1.0,
    use_mjx: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Create quadruped env (flat, rough, push, go2).

    Uses robot registry for model path. When use_mjx=True, creates MJX env
    for JAX-based planning (MBD, 2GO). Falls back to MuJoCo physics when use_mjx=False.
    """
    from genedynamics.envs.factories import make_env
    from genedynamics.robots import get_robot_registry

    if env_name is None:
        if use_mjx:
            env_name = "quadruped_flat_mjx" if (model or "ant") in ("ant", "flat", "rough", "push") else "quadruped_go2_mjx"
        else:
            reg = get_robot_registry()
            fn = reg.get_env_factory_name("quadruped", model or "flat")
            env_name = fn or "quadruped_flat_physics"

    if use_mjx and "mjx" not in env_name:
        if (model or "ant") in ("go2",):
            env_name = "quadruped_go2_mjx"
        else:
            env_name = "quadruped_flat_mjx"

    kw = dict(dt=dt, horizon=horizon, target=target, control_limit=control_limit, **kwargs)
    if model is not None:
        kw["model"] = model
    return make_env(env_name, **kw)


def make_quadruped_energy(
    target: tuple = (2.0, 0.0, 0.5),
    env_name: str = "quadruped_flat_mjx",
) -> Any:
    """Create energy functional for quadruped (MBD, 2GO)."""
    from genedynamics.envs.factories import make_energy

    return make_energy(env_name)


def make_humanoid_energy(
    target: tuple = (3.0, 0.0, 1.0),
    env_name: str = "humanoid_simplified_mjx",
) -> Any:
    """Create energy functional for humanoid (MBD, 2GO)."""
    from genedynamics.envs.factories import make_energy

    return make_energy(env_name)


def make_mbd_planner(
    env: Any,
    energy: Any,
    horizon: int = 64,
    dt: Optional[float] = None,
    seed: int = 0,
    obstacles: Optional[Any] = None,
    env_name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create MBD planner for quadruped or other env with jax_transition.

    Args:
        env: Environment with transition, jax_transition, target, control_limit
        energy: LegacyEnergyFunctional (from make_quadruped_energy or make_energy)
        horizon: Planning horizon
        dt: Time step (default: env.dt)
        seed: Random seed
        obstacles: Optional obstacle manager
        env_name: For task_spec (e.g. "quadruped_flat_mjx")
        **kwargs: Nsample, Ndiffuse, temp_sample, etc.
    """
    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.core.dynamics.adapters import EnvDynamicsAdapter
    from genedynamics.core.task_spec import get_default_task_spec
    from genedynamics.solvers.single.mbd import MBDSolver

    backend = RuntimeBackendManager.get_backend()
    dynamics = EnvDynamicsAdapter(env)
    dt_val = dt if dt is not None else getattr(env, "dt", 0.05)
    task_spec = get_default_task_spec(env_plugin=None, env_name=env_name or "quadruped_flat_mjx")

    solver = MBDSolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=horizon,
        dt=dt_val,
        Nsample=int(kwargs.get("Nsample", 1024)),
        Ndiffuse=int(kwargs.get("Ndiffuse", 100)),
        temp_sample=float(kwargs.get("temp_sample", 0.1)),
        beta0=float(kwargs.get("beta0", 1e-4)),
        betaT=float(kwargs.get("betaT", 1e-2)),
        action_limit=float(getattr(env, "control_limit", 1.0)),
        action_extra_sigma=float(kwargs.get("action_extra_sigma", 0.0)),
        seed=seed,
        scheduler=kwargs.get("scheduler"),
        show_tqdm=bool(kwargs.get("show_tqdm", False)),
        num_modes=int(kwargs.get("num_modes", 1)),
        mode_strategy=str(kwargs.get("mode_strategy", "multirun")),
        diversity_eta=float(kwargs.get("diversity_eta", 1.0)),
        diversity_topK_cand=kwargs.get("diversity_topK_cand"),
        diversity_use_state=bool(kwargs.get("diversity_use_state", True)),
        terminal_energy_weight=float(kwargs.get("terminal_energy_weight", 100.0)),
        position_extractor=task_spec.extract_position,
        position_dim=task_spec.position_dim,
    )
    return solver


def make_cfsmbd_full_planner(
    env: Any,
    energy: Any,
    horizon: int = 64,
    obstacles: Optional[Any] = None,
    use_mjx: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Create CFS-MBD full trajectory QP planner (MD-COAS) for quadruped.

    Uses CFSQPFullFilter for CFS-based action projection. Supports 3D
    position (quadruped base) via position_extractor and position_dim.

    Args:
        env: Quadruped environment (MJX for JAX planning)
        energy: LegacyEnergyFunctional
        horizon: Planning horizon
        obstacles: Optional ObstacleManager for obstacle_avoid
        use_mjx: If True, env is MJX
        **kwargs: Nsample, Ndiffuse, max_constraints_per_point, etc.
    """
    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.core.dynamics import DynamicsToEnvAdapter
    from genedynamics.core.task_spec import get_default_task_spec
    from genedynamics.core.constraints.action_filters.cfs_qp_full import CFSQPFullFilter
    from genedynamics.solvers.single.cfsmbd import CFSMBDSolver

    backend = RuntimeBackendManager.get_backend()
    dynamics = DynamicsToEnvAdapter(env, dt=kwargs.get("dt", getattr(env, "dt", 0.05)))
    env_name = "quadruped_flat_mjx" if use_mjx else "quadruped_flat_physics"
    if "go2" in str(getattr(env, "model", "")).lower():
        env_name = "quadruped_go2_mjx" if use_mjx else "quadruped_go2_physics"
    task_spec = get_default_task_spec(env_plugin=None, env_name=env_name)

    convexifier_name = str(kwargs.get("cfs_action_convexifier", "cfs_action"))
    constraint_filter = CFSQPFullFilter(
        max_constraints_per_point=int(kwargs.get("max_constraints_per_point", 8)),
        constraint_margin=float(kwargs.get("constraint_margin", 0.25)),
        use_slack=False,
        convexifier_name=convexifier_name,
        position_extractor=task_spec.extract_position,
        position_dim=task_spec.position_dim,
    )

    dt_val = kwargs.get("dt", getattr(env, "dt", 0.05))
    return CFSMBDSolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=horizon,
        dt=dt_val,
        Nsample=int(kwargs.get("Nsample", 512)),
        Ndiffuse=int(kwargs.get("Ndiffuse", 100)),
        temp_sample=float(kwargs.get("temp_sample", 0.3)),
        beta0=float(kwargs.get("beta0", 1e-4)),
        betaT=float(kwargs.get("betaT", 1e-2)),
        action_limit=float(getattr(env, "control_limit", 1.0)),
        seed=int(kwargs.get("seed", 0)),
        scheduler=kwargs.get("scheduler"),
        constraint_filter=constraint_filter,
        obstacles=obstacles,
        show_tqdm=bool(kwargs.get("show_tqdm", False)),
        aug_lambda=float(kwargs.get("aug_lambda", 0.0)),
        aug_rho=float(kwargs.get("aug_rho", 1.0)),
        action_extra_sigma=float(kwargs.get("action_extra_sigma", 0.0)),
        num_modes=int(kwargs.get("num_modes", 1)),
        mode_strategy=str(kwargs.get("mode_strategy", "multirun")),
        diversity_eta=float(kwargs.get("diversity_eta", 1.0)),
        diversity_topK_cand=kwargs.get("diversity_topK_cand"),
        diversity_use_state=bool(kwargs.get("diversity_use_state", True)),
        position_extractor=task_spec.extract_position,
        position_dim=task_spec.position_dim,
    )


def make_cfsmbd_planner(
    env: Any,
    energy: Any,
    horizon: int = 64,
    obstacles: Optional[Any] = None,
    use_mjx: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Create CFS-MBD per-step QP planner for quadruped.

    Uses CFSQPPerStepFilter. Same interface as make_cfsmbd_full_planner.
    """
    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.core.dynamics import DynamicsToEnvAdapter
    from genedynamics.core.task_spec import get_default_task_spec
    from genedynamics.core.constraints.action_filters.cfs_qp_perstep import CFSQPPerStepFilter
    from genedynamics.solvers.single.cfsmbd import CFSMBDSolver

    backend = RuntimeBackendManager.get_backend()
    dynamics = DynamicsToEnvAdapter(env, dt=kwargs.get("dt", getattr(env, "dt", 0.05)))
    env_name = "quadruped_flat_mjx" if use_mjx else "quadruped_flat_physics"
    if "go2" in str(getattr(env, "model", "")).lower():
        env_name = "quadruped_go2_mjx" if use_mjx else "quadruped_go2_physics"
    task_spec = get_default_task_spec(env_plugin=None, env_name=env_name)

    convexifier_name = str(kwargs.get("cfs_action_convexifier", "cfs_action"))
    constraint_filter = CFSQPPerStepFilter(
        max_constraints_per_point=int(kwargs.get("max_constraints_per_point", 8)),
        constraint_margin=float(kwargs.get("constraint_margin", 0.25)),
        use_slack=False,
        convexifier_name=convexifier_name,
        position_extractor=task_spec.extract_position,
        position_dim=task_spec.position_dim,
    )

    dt_val = kwargs.get("dt", getattr(env, "dt", 0.05))
    return CFSMBDSolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=horizon,
        dt=dt_val,
        Nsample=int(kwargs.get("Nsample", 512)),
        Ndiffuse=int(kwargs.get("Ndiffuse", 100)),
        temp_sample=float(kwargs.get("temp_sample", 0.3)),
        beta0=float(kwargs.get("beta0", 1e-4)),
        betaT=float(kwargs.get("betaT", 1e-2)),
        action_limit=float(getattr(env, "control_limit", 1.0)),
        seed=int(kwargs.get("seed", 0)),
        scheduler=kwargs.get("scheduler"),
        constraint_filter=constraint_filter,
        obstacles=obstacles,
        show_tqdm=bool(kwargs.get("show_tqdm", False)),
        aug_lambda=float(kwargs.get("aug_lambda", 0.0)),
        aug_rho=float(kwargs.get("aug_rho", 1.0)),
        position_extractor=task_spec.extract_position,
        position_dim=task_spec.position_dim,
    )


def make_quadruped_planner(
    env: Any,
    horizon: int = 64,
    planner_type: str = "stand",
    energy: Optional[Any] = None,
    use_mjx: bool = False,
    obstacles: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Create quadruped planner.

    Args:
        env: Quadruped environment
        horizon: Planning horizon
        planner_type: "stand" | "mbd" | "cfsmbd" | "cfsmbd_full" (MD-COAS)
        energy: Required when planner_type in ("mbd", "cfsmbd", "cfsmbd_full")
        use_mjx: If True and planner_type=mbd/cfsmbd, env should be MJX
        obstacles: Optional ObstacleManager for cfsmbd/cfsmbd_full
        **kwargs: Passed to planner factory
    """
    if planner_type == "mbd":
        if energy is None:
            energy = make_quadruped_energy(
                target=tuple(getattr(env, "target", (2.0, 0.0, 0.5))),
                env_name="quadruped_flat_mjx",
            )
        return make_mbd_planner(
            env=env,
            energy=energy,
            horizon=horizon,
            dt=getattr(env, "dt", 0.05),
            env_name="quadruped_flat_mjx" if use_mjx or "mjx" in str(type(env).__name__).lower() else "quadruped_flat_physics",
            **kwargs,
        )
    if planner_type == "cfsmbd_full":
        if energy is None:
            energy = make_quadruped_energy(
                target=tuple(getattr(env, "target", (2.0, 0.0, 0.5))),
                env_name="quadruped_flat_mjx",
            )
        return make_cfsmbd_full_planner(
            env=env,
            energy=energy,
            horizon=horizon,
            obstacles=obstacles,
            use_mjx=use_mjx,
            **kwargs,
        )
    if planner_type == "cfsmbd":
        if energy is None:
            energy = make_quadruped_energy(
                target=tuple(getattr(env, "target", (2.0, 0.0, 0.5))),
                env_name="quadruped_flat_mjx",
            )
        return make_cfsmbd_planner(
            env=env,
            energy=energy,
            horizon=horizon,
            obstacles=obstacles,
            use_mjx=use_mjx,
            **kwargs,
        )
    from genedynamics.deploy.quadruped_planner import QuadrupedStandPlanner

    return QuadrupedStandPlanner(
        act_dim=getattr(env, "act_dim", 8),
        horizon=horizon,
    )


def make_humanoid_env(
    model: Optional[str] = None,
    env_name: Optional[str] = None,
    dt: float = 0.05,
    horizon: int = 200,
    target: tuple = (3.0, 0.0, 1.0),
    control_limit: float = 0.4,
    use_mjx: bool = False,
    obstacles: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Create humanoid env. Uses robot registry for env_factory and model path.
    When use_mjx=True, creates MJX env for JAX-based planning (MBD, 2GO).
    """
    from genedynamics.envs.factories import make_env
    from genedynamics.robots import get_robot_registry

    kw = dict(dt=dt, horizon=horizon, target=target, control_limit=control_limit, **kwargs)
    if model is not None:
        kw["model"] = model
    if obstacles is not None:
        kw["obstacles"] = obstacles

    if env_name is None:
        if use_mjx:
            env_name = "humanoid_g1_mjx" if (model or "humanoid") == "g1" else "humanoid_simplified_mjx"
        else:
            env_name = "humanoid_simplified_physics"
            if model:
                reg = get_robot_registry()
                fn = reg.get_env_factory_name("humanoid", model)
                if fn:
                    env_name = fn

    if use_mjx and "mjx" not in env_name:
        env_name = "humanoid_g1_mjx" if (model or "") == "g1" else "humanoid_simplified_mjx"

    return make_env(env_name, **kw)


def make_humanoid_planner(
    env: Any,
    horizon: int = 64,
    planner_type: str = "stand",
    energy: Optional[Any] = None,
    use_mjx: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Create humanoid planner.

    Args:
        env: Humanoid environment
        horizon: Planning horizon
        planner_type: "stand" (zero actions) | "mbd" (MBD diffusion)
        energy: Required when planner_type="mbd"
        use_mjx: If True and planner_type=mbd, env should be MJX
        **kwargs: Passed to make_mbd_planner when planner_type="mbd"
    """
    if planner_type == "mbd":
        if energy is None:
            energy = make_humanoid_energy(
                target=tuple(getattr(env, "target", (3.0, 0.0, 1.0))),
                env_name="humanoid_simplified_mjx",
            )
        env_name = "humanoid_g1_mjx" if getattr(env, "model", "") == "g1" else "humanoid_simplified_mjx"
        return make_mbd_planner(
            env=env,
            energy=energy,
            horizon=horizon,
            dt=getattr(env, "dt", 0.05),
            env_name=env_name,
            **kwargs,
        )
    from genedynamics.deploy.quadruped_planner import HumanoidStandPlanner

    return HumanoidStandPlanner(
        act_dim=getattr(env, "act_dim", 17),
        horizon=horizon,
    )


def make_2go_planner(
    env: Any,
    energy: Any,
    obstacles: Optional[Any] = None,
    horizon: int = 64,
    dt: float = 0.05,
    seed: int = 0,
    scheduler: Any = None,
    **kwargs: Any,
) -> Any:
    """Create 2GO planner for UAV 3D."""
    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.core.dynamics import DynamicsToEnvAdapter
    from genedynamics.core.task_spec import get_default_task_spec
    from genedynamics.core.constraints.action_filters.cfs_qp_full import CFSQPFullFilter
    from genedynamics.solvers.single.twogo import TwoGOSolver

    backend = RuntimeBackendManager.get_backend()
    dynamics = DynamicsToEnvAdapter(env, dt=dt)
    task_spec = get_default_task_spec(env_name="drone_full_3d_physics")

    constraint_filter = CFSQPFullFilter(
        max_constraints_per_point=int(kwargs.get("max_constraints_per_point", 8)),
        constraint_margin=float(kwargs.get("constraint_margin", 0.25)),
        use_slack=False,
        convexifier_name="cfs_action",
        position_extractor=task_spec.extract_position,
    )

    _exclude = ("max_constraints_per_point", "constraint_margin", "Nsample", "Ndiffuse", "temp_sample", "beta0", "betaT")
    extra_kw = {k: v for k, v in kwargs.items() if k not in _exclude}
    return TwoGOSolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=horizon,
        dt=dt,
        Nsample=int(kwargs.get("Nsample", 256)),
        Ndiffuse=int(kwargs.get("Ndiffuse", 100)),
        temp_sample=float(kwargs.get("temp_sample", 0.3)),
        beta0=float(kwargs.get("beta0", 1e-4)),
        betaT=float(kwargs.get("betaT", 1e-2)),
        action_limit=float(getattr(env, "control_limit", 1.0)),
        seed=seed,
        scheduler=scheduler,
        constraint_filter=constraint_filter,
        obstacles=obstacles,
        position_extractor=task_spec.extract_position,
        position_dim=task_spec.position_dim,
        **extra_kw,
    )


def _infer_task_spec(env: Any) -> tuple:
    """Infer (spec, nq, nv, act_dim) from env."""
    act_dim = getattr(env, "act_dim", 4)
    nq = getattr(env, "nq", None)
    nv = getattr(env, "nv", None)
    state_dim = getattr(env, "state_dim", None)

    # Quadruped: nq=15, nv=14; Humanoid: nq=24, nv=23
    if nq is not None and nv is not None and nq >= 10:
        from genedynamics.tasks.quadruped.spec import QuadrupedTaskSpec
        spec = QuadrupedTaskSpec(
            nq=nq, nv=nv, act_dim=act_dim,
            target=tuple(getattr(env, "target", (2.0, 0.0, 0.5))),
            control_limit=float(getattr(env, "control_limit", 1.0)),
        )
        return spec, nq, nv, act_dim
    # UAV 3D default
    from genedynamics.tasks.uav3d.spec import UAV3DTaskSpec
    spec = UAV3DTaskSpec(
        target=tuple(getattr(env, "target", (0.0, 0.0, 1.0))),
        control_limit=float(getattr(env, "control_limit", 1.0)),
    )
    nq = nq if nq is not None else 6
    nv = nv if nv is not None else 6
    return spec, nq, nv, act_dim


def make_executor(
    env: Any,
    planner: Any,
    config: Dict[str, Any],
    episode_dir: Optional[str] = None,
) -> Any:
    """Create executor with all components."""
    from genedynamics.execution.core.contracts import ExecutionMode, SessionConfig
    from genedynamics.execution.core.safety import SafetyGuard
    from genedynamics.execution.core.executor import Executor
    from genedynamics.execution.bridges.planner_bridge import PlannerBridge
    from genedynamics.execution.providers.sim_state_provider import SimStateProvider
    from genedynamics.execution.publishers.sim_control_publisher import SimControlPublisher
    from genedynamics.execution.logging.telemetry import TelemetryLogger
    from genedynamics.execution.logging.episode_writer import EpisodeWriter

    spec, nq, nv, act_dim = _infer_task_spec(env)

    state_provider = SimStateProvider(env, nq=nq, nv=nv)
    control_publisher = SimControlPublisher(env)
    planner_bridge = PlannerBridge(planner, horizon=config.get("horizon", 64), plan_mode="plan_once")

    ctrl_lim = float(getattr(env, "control_limit", 1.0))
    # Quadruped: torque [-lim, +lim]; UAV: MBD/2go output [-lim, +lim], env maps to thrust [0, lim]
    action_min = np.full(act_dim, -ctrl_lim, dtype=np.float32)
    action_max = np.full(act_dim, ctrl_lim, dtype=np.float32)
    safety_guard = SafetyGuard(
        action_clip_min=action_min,
        action_clip_max=action_max,
        fallback_action=np.zeros(act_dim, dtype=np.float32),
    )

    mode_str = config.get("execution_mode", "sim_only")
    try:
        mode = ExecutionMode(mode_str)
    except ValueError:
        mode = ExecutionMode.SIM_ONLY

    session_config = SessionConfig(
        execution_mode=mode,
        control_rate_hz=float(config.get("control_rate_hz", 20.0)),
        plan_rate_hz=float(config.get("plan_rate_hz", 10.0)),
        real_time_factor=float(config.get("real_time_factor", 1.0)),
        sync_mode=bool(config.get("sync_mode", True)),
        record=bool(config.get("record", True)),
        episode_dir=episode_dir,
        action_clip_min=action_min,
        action_clip_max=action_max,
        tags=config.get("tags", {}),
        extra=config,
    )

    episode_writer = None
    if episode_dir:
        episode_writer = EpisodeWriter(episode_dir, tags=session_config.tags)

    telemetry = TelemetryLogger(enabled=session_config.record)

    return Executor(
        state_provider=state_provider,
        control_publisher=control_publisher,
        planner_bridge=planner_bridge,
        safety_guard=safety_guard,
        config=session_config,
        telemetry=telemetry,
        episode_writer=episode_writer,
    )
