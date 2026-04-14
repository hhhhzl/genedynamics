"""
SoftZoo adapters: env creation, morphology/controller encoding, rollout decoding.

Industrial-grade adapters with lazy imports, path resolution, and
extensible design for future soft robot tasks.
"""

from __future__ import annotations

import contextlib
import io
import os
import platform
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .bootstrap import ensure_softzoo_on_path, get_softzoo_paths
from .config import SoftZooEnvConfig, SoftZooRuntimeConfig
from .schemas import (
    SoftZooTaskSpec,
    SoftZooFidelitySpec,
    SoftZooModeSpec,
    SoftZooRolloutResult,
    SoftZooRolloutBatchResult,
)
from .task_registry import get_task_spec


def _apply_mode_friction_to_cfg(cfg: Any, friction: float) -> None:
    """Set terrain friction in config. Modifies cfg in-place."""
    items = getattr(getattr(cfg, "ENVIRONMENT", None), "ITEMS", None)
    if items is None:
        return
    for item in items:
        if hasattr(item, "type") and "Terrain" in str(getattr(item, "type", "")):
            setattr(item, "friction", float(friction))
            return


def _set_cfg_attr(cfg: Any, key: str, val: Any) -> None:
    """Set nested yacs config attr by dotted key."""
    fields = key.split(".")
    ptr = cfg
    for f in fields[:-1]:
        ptr = getattr(ptr, f)
    setattr(ptr, fields[-1], val)


def _cfg_to_plain_dict(obj: Any) -> Any:
    """
    Recursively convert CfgNode to plain dict (str, int, float, list, dict, bool, None).
    Ensures YAML output is loadable by yaml.safe_load (avoids tag:yaml.org,2002 ConstructorError).
    """
    from yacs.config import CfgNode as CN

    if isinstance(obj, CN):
        return {k: _cfg_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cfg_to_plain_dict(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # numpy/torch scalars, etc.
    if hasattr(obj, "item"):
        return float(obj.item()) if hasattr(obj, "item") else float(obj)
    return obj


def _init_taichi(runtime_config: SoftZooRuntimeConfig) -> None:
    """
    Initialize Taichi with platform-safe arch selection.
    Priority: explicit runtime_config.ti_arch / env override > platform defaults.
    """
    import taichi as ti

    arch_name = (runtime_config.ti_arch or os.environ.get("SOFTZOO_TI_ARCH") or "").lower()
    arch_map = {
        "cpu": ti.cpu,
        "metal": ti.metal,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
    }

    if arch_name in arch_map:
        preferred = [arch_map[arch_name]]
    else:
        is_macos = platform.system() == "Darwin"
        # macOS: prefer CPU for stability, then try Metal.
        if is_macos:
            preferred = [ti.cpu, ti.metal]
        else:
            wants_gpu = runtime_config.device in ("cuda", "torch_gpu")
            preferred = [ti.cuda, ti.cpu] if wants_gpu else [ti.cpu]

    init_errors = []
    for arch in preferred:
        try:
            kwargs: Dict[str, Any] = {
                "arch": arch,
                "default_fp": ti.f32,
                "default_ip": ti.i32,
            }
            if arch == ti.cuda and runtime_config.ti_device_memory_fraction is not None:
                kwargs["device_memory_fraction"] = runtime_config.ti_device_memory_fraction
            ti.init(**kwargs)
            return
        except Exception as e:
            init_errors.append(f"{arch}: {type(e).__name__}: {e}")

    raise RuntimeError("Taichi init failed for all candidate backends: " + " | ".join(init_errors))


def make_softzoo_env(
    task_spec: Optional[SoftZooTaskSpec] = None,
    task_id: Optional[str] = None,
    fidelity_spec: Optional[SoftZooFidelitySpec] = None,
    mode_spec: Optional[SoftZooModeSpec] = None,
    runtime_config: Optional[SoftZooRuntimeConfig] = None,
    env_config: Optional[SoftZooEnvConfig] = None,
) -> Any:
    """
    Create a SoftZoo environment instance.

    Args:
        task_spec: Full task spec (takes precedence over task_id)
        task_id: Task id to lookup from registry
        fidelity_spec: Fidelity overrides
        runtime_config: Runtime config (paths, device, etc.)
        env_config: Env creation overrides

    Returns:
        SoftZoo env instance (LandEnvironment, AquaticEnvironment, etc.)
    """
    if runtime_config is None:
        runtime_config = SoftZooRuntimeConfig()

    ensure_softzoo_on_path(runtime_config.project_root if runtime_config else None)
    paths = get_softzoo_paths(runtime_config.project_root if runtime_config else None)
    # Ensure Taichi is initialized with platform-aware backend selection.
    _init_taichi(runtime_config)
    if platform.system() == "Darwin":
        # Work around Taichi kernel crashes seen on macOS in occupancy setup.
        os.environ.setdefault("SOFTZOO_SKIP_OCCUPANCY_KERNELS", "1")
    elif (runtime_config.ti_arch or "").lower() == "cuda":
        # Taichi 1.7.x + CUDA: occupancy kernels trigger device-side assert.
        os.environ.setdefault("SOFTZOO_SKIP_OCCUPANCY_KERNELS", "1")

    if task_spec is None:
        task_spec = get_task_spec(task_id or "crawling_ground")
    if env_config is not None:
        cfg_file = env_config.env_config_file
        pcd_name = env_config.pcd_name
        pcd_path = env_config.pcd_path
    else:
        cfg_file = task_spec.env_config_file
        pcd_name = task_spec.robot.morphology.pcd_name
        pcd_path = task_spec.robot.morphology.pcd_path

    # Resolve PCD path
    if pcd_path is None or not os.path.isabs(pcd_path):
        resolved_pcd = paths.resolve_pcd_path(pcd_name.replace(".pcd", ""))
        if not resolved_pcd.exists():
            pkg_pcd = paths.package_dir / "assets" / "meshes" / "pcd" / f"{pcd_name}.pcd"
            if pkg_pcd.exists():
                resolved_pcd = pkg_pcd
            else:
                resolved_pcd = paths.meshes_pcd_dir / f"{pcd_name}.pcd"
        pcd_path = str(resolved_pcd)

    cfg_kwargs: Dict[str, Any] = {
        "ENVIRONMENT.use_renderer": runtime_config.use_renderer,
    }
    if fidelity_spec is not None:
        cfg_kwargs["SIMULATOR.max_substeps"] = fidelity_spec.max_substeps
        cfg_kwargs["SIMULATOR.max_substeps_local"] = fidelity_spec.max_substeps_local
        cfg_kwargs["SIMULATOR.quality"] = fidelity_spec.quality
        # TaichiSim: max_steps = max_substeps // n_substeps; MoveForward asserts max_episode_steps <= max_steps
        frame_dt = 8e-3
        default_dt = 5e-4
        n_substeps = int(frame_dt / default_dt) + 1
        max_steps = fidelity_spec.max_substeps // n_substeps
        cfg_kwargs["ENVIRONMENT.objective_config.max_episode_steps"] = min(
            200, max(1, max_steps)
        )
    if env_config and env_config.cfg_kwargs:
        cfg_kwargs.update(env_config.cfg_kwargs)

    # Mode override: build full cfg with terrain friction when mode_spec provided
    cfg_file_or_cfg: Any = cfg_file
    _temp_cfg_file: Optional[str] = None  # noqa: F841
    if mode_spec is not None:
        ensure_softzoo_on_path(runtime_config.project_root)
        from softzoo.configs.config import get_cfg_defaults

        cfg_path = os.path.join(paths.env_configs_dir, cfg_file)
        cfg = get_cfg_defaults()
        cfg.merge_from_file(cfg_path)
        for k, v in cfg_kwargs.items():
            _set_cfg_attr(cfg, k, v)
        _apply_mode_friction_to_cfg(cfg, mode_spec.friction)
        import yaml

        plain = _cfg_to_plain_dict(cfg)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.safe_dump(plain, f, default_flow_style=False, sort_keys=False)
            _temp_cfg_file = f.name
        cfg_file_or_cfg = os.path.basename(_temp_cfg_file)
        # Patch ENV_CONFIGS_DIR to the temp dir so env finds our config
        _temp_dir = os.path.dirname(_temp_cfg_file)
        import softzoo.envs as envs_mod
        envs_mod.ENV_CONFIGS_DIR = _temp_dir

    device = runtime_config.device
    if device == "torch_gpu":
        device = "torch_gpu"
    elif device == "cuda":
        device = "torch_gpu"
    elif device == "numpy":
        device = "numpy"
    else:
        device = "torch_cpu"

    out_dir = runtime_config.out_dir
    # Pass None so BaseEnv skips ti.init (we already initialized above)
    ti_frac = None

    env_type = task_spec.env_type.value if hasattr(task_spec.env_type, "value") else str(task_spec.env_type)

    if env_type == "land_environment":
        from softzoo.envs.land_environment import LandEnvironment
        env_cls = LandEnvironment
    elif env_type == "aquatic_environment":
        from softzoo.envs.aquatic_environment import AquaticEnvironment
        env_cls = AquaticEnvironment
    elif env_type == "dummy_env":
        from softzoo.envs.dummy_env import DummyEnv
        env_cls = DummyEnv
    elif env_type == "manipulation_environment":
        from softzoo.envs.manipulation_environment import ManipulationEnvironment
        env_cls = ManipulationEnvironment
    else:
        from softzoo.envs.land_environment import LandEnvironment
        env_cls = LandEnvironment

    # Patch ENV_CONFIGS_DIR so softzoo finds configs. `from . import ENV_CONFIGS_DIR`
    # in submodules caches the value at import time, so update both package and submodule
    # namespaces to keep them in sync across repeated make_softzoo_env calls.
    import softzoo.envs as envs_mod
    _env_cfg_dir = (
        os.path.dirname(_temp_cfg_file) if _temp_cfg_file is not None else str(paths.env_configs_dir)
    )
    envs_mod.ENV_CONFIGS_DIR = _env_cfg_dir
    for _sub in ("land_environment", "aquatic_environment", "manipulation_environment", "dummy_env"):
        _m = sys.modules.get(f"softzoo.envs.{_sub}")
        if _m is not None and hasattr(_m, "ENV_CONFIGS_DIR"):
            _m.ENV_CONFIGS_DIR = _env_cfg_dir

    env = env_cls(
        cfg_file=cfg_file_or_cfg,
        out_dir=out_dir,
        device=device,
        cfg_kwargs=cfg_kwargs,
        ti_device_memory_fraction=ti_frac,
        initialize=False,
    )

    if runtime_config.suppress_init_print:
        with _suppress_stdout():
            env.initialize()
    else:
        env.initialize()

    # Attach metadata for adapter use
    env._softzoo_pcd_path = pcd_path
    env._softzoo_task_spec = task_spec
    env._softzoo_fidelity_spec = fidelity_spec
    return env


@contextlib.contextmanager
def _suppress_stdout():
    """Temporarily suppress stdout."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def encode_morphology(
    x: np.ndarray,
    task_spec: SoftZooTaskSpec,
    *,
    designer_type: Optional[str] = None,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Encode morphology parameters x into SoftZoo design dict.

    For annotated_pcd, x can be empty or scaling multipliers.
    For voxel/sdf, x is the design coefficients.

    Args:
        x: Morphology parameters (flattened)
        task_spec: Task spec
        designer_type: Override designer type
        env: Optional env, used to expand scalar multipliers to tensor shapes

    Returns:
        design dict for env.reset(design)
    """
    dtype = designer_type or task_spec.robot.morphology.designer_type.value
    design: Dict[str, Any] = {}

    if dtype == "annotated_pcd":
        x = np.asarray(x if x is not None else np.array([]), dtype=np.float32).ravel()
        geom_mul = (
            float(x[0])
            if x.size >= 1 and x[0] > 0
            else float(task_spec.robot.morphology.passive_geometry_mul)
        )
        soft_mul = float(x[1]) if x.size >= 2 and x[1] > 0 else 1.0
        act_mul = float(x[2]) if x.size >= 3 and x[2] > 0 else 1.0

        if env is None:
            # Backward-compatible fallback (legacy callers without env).
            design["geometry"] = geom_mul
            design["softness"] = soft_mul
            design["actuator"] = act_mul
        else:
            n_particles = int(env.design_space.n_particles)
            n_actuators = int(env.design_space.n_actuators)
            geometry = np.full((n_particles,), geom_mul, dtype=np.float32)
            softness = np.full((n_particles,), soft_mul, dtype=np.float32)
            actuator = np.full(
                (n_actuators, n_particles),
                act_mul / max(n_actuators, 1),
                dtype=np.float32,
            )
            if "Torch" in str(getattr(env.sim, "device", "")):
                import torch
                design["geometry"] = torch.from_numpy(geometry)
                design["softness"] = torch.from_numpy(softness)
                design["actuator"] = torch.from_numpy(actuator)
            else:
                design["geometry"] = geometry
                design["softness"] = softness
                design["actuator"] = actuator
    elif dtype in ("voxel", "vbr", "sdf_basis", "diff_cppn", "wass_barycenter"):
        # Coefficient-based: x maps to design coefficients
        if x is not None and x.size > 0:
            design["geometry"] = x
            design["softness"] = x if "softness" in design else None
            design["actuator"] = x if "actuator" in design else None
    return design


def encode_controller(
    phi: np.ndarray,
    task_spec: SoftZooTaskSpec,
    env: Any,
    *,
    controller_type: str = "sin_wave_open_loop",
) -> Any:
    """
    Create a SoftZoo controller from parameters phi.

    For sin_wave_open_loop, phi = [omega_1, ..., omega_k, amp_1, ..., amp_k].
    Default: first n_actuators as omega, rest as amplitudes.

    Args:
        phi: Controller parameters
        task_spec: Task spec
        env: SoftZoo env (for n_actuators, obs_space)
        controller_type: Controller type

    Returns:
        Controller instance
    """
    ensure_softzoo_on_path()
    n_actuators = env.design_space.n_actuators
    actuation_strength = env.cfg.ENVIRONMENT.actuation_strength
    device = "cuda" if hasattr(env.sim, "device") and str(env.sim.device) == "cuda" else "cpu"

    phi = np.asarray(phi, dtype=np.float64).ravel()
    if phi.size == 0:
        phi = np.array([30.0] * min(4, n_actuators), dtype=np.float64)

    if controller_type == "sin_wave_open_loop":
        from algorithms.controllers.sinwave import SinWaveOpenLoop
        n_sin = min(len(phi) // 2, 4, n_actuators)
        omega = phi[:n_sin].tolist() if n_sin > 0 else [30.0]
        if len(omega) < 2:
            omega = [float(omega[0])] * 2
        ctrl = SinWaveOpenLoop(
            n_actuators=n_actuators,
            actuation_strength=actuation_strength,
            lr=0.1,
            device=device,
            n_sin_waves=n_sin,
            actuation_omega=omega,
        )
    else:
        from algorithms.controllers.sinwave import SinWaveOpenLoop
        ctrl = SinWaveOpenLoop(
            n_actuators=n_actuators,
            actuation_strength=actuation_strength,
            lr=0.1,
            device=device,
            n_sin_waves=4,
            actuation_omega=[30.0, 100.0],
        )
    return ctrl


def decode_rollout(
    obs: Any,
    reward: float,
    done: bool,
    info: Dict[str, Any],
    env: Any,
    *,
    total_return: Optional[float] = None,
    num_steps: Optional[int] = None,
    wall_time: float = 0.0,
) -> SoftZooRolloutResult:
    """
    Decode raw env step output into SoftZooRolloutResult.

    Args:
        obs: Last observation
        reward: Last step reward
        done: Done flag
        info: Info dict from env.step
        env: SoftZoo env (for final COM extraction)
        total_return: Precomputed episode return
        num_steps: Precomputed step count
        wall_time: Wall time for rollout

    Returns:
        SoftZooRolloutResult
    """
    if total_return is None:
        total_return = float(reward)
    if num_steps is None:
        num_steps = info.get("frame_idx", 0)

    final_com = np.zeros(3, dtype=np.float32)
    if obs is not None and isinstance(obs, dict) and "com" in obs:
        com = obs["com"]
        if hasattr(com, "numpy"):
            final_com = np.asarray(com.numpy(), dtype=np.float32).ravel()[:3]
        else:
            final_com = np.asarray(com, dtype=np.float32).ravel()[:3]
    elif hasattr(env, "design_space") and hasattr(env, "sim") and hasattr(env.sim, "solver"):
        try:
            s = env.sim.solver.current_s
            robot_x = env.design_space.get_x(s)
            if hasattr(robot_x, "mean"):
                mx = robot_x.mean(0)
                if hasattr(mx, "numpy"):
                    final_com = np.asarray(mx.numpy(), dtype=np.float32).ravel()[:3]
                else:
                    final_com = np.asarray(mx, dtype=np.float32).ravel()[:3]
        except Exception:
            pass

    success = bool(done and total_return > 0)
    return SoftZooRolloutResult(
        return_=float(total_return),
        success=success,
        num_steps=int(num_steps),
        final_com=final_com,
        trajectory=info.get("trajectory"),
        failure_code=info.get("failure_code"),
        wall_time=wall_time,
        extra=dict(info),
    )
