"""
Robot registry for multi-backend support.

Register quadruped/humanoid models with env class, spec, and model path.
Enables adding new robots (Go2, A1, H1, etc.) without scattering logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

_registry: Optional["RobotRegistry"] = None


@dataclass
class RobotEntry:
    """Single robot model entry."""

    robot_type: str  # "quadruped" | "humanoid"
    model_id: str  # "ant", "go2", "humanoid", "h1"
    env_factory_name: str  # e.g. "quadruped_flat_physics"
    nq: int
    nv: int
    act_dim: int
    model_path_resolver: Optional[Callable[[], str]] = None  # () -> path to MJCF
    spec_class: Optional[Type] = None
    description: str = ""


class RobotRegistry:
    """
    Registry for robot models (quadruped, humanoid).

    Usage:
        registry = get_robot_registry()
        registry.register_quadruped("go2", env_factory="quadruped_go2_physics", ...)
        entry = registry.get("quadruped", "go2")
    """

    def __init__(self):
        self._entries: Dict[tuple, RobotEntry] = {}

    def register(
        self,
        robot_type: str,
        model_id: str,
        env_factory_name: str,
        nq: int,
        nv: int,
        act_dim: int,
        model_path_resolver: Optional[Callable[[], str]] = None,
        spec_class: Optional[Type] = None,
        description: str = "",
    ) -> None:
        """Register a robot model."""
        self._entries[(robot_type, model_id)] = RobotEntry(
            robot_type=robot_type,
            model_id=model_id,
            env_factory_name=env_factory_name,
            nq=nq,
            nv=nv,
            act_dim=act_dim,
            model_path_resolver=model_path_resolver,
            spec_class=spec_class,
            description=description,
        )

    def get(self, robot_type: str, model_id: str) -> Optional[RobotEntry]:
        """Get entry by robot_type and model_id."""
        return self._entries.get((robot_type, model_id))

    def list_models(self, robot_type: str) -> List[str]:
        """List registered model IDs for a robot type."""
        return [e.model_id for (rt, mid), e in self._entries.items() if rt == robot_type]

    def get_env_factory_name(self, robot_type: str, model_id: str) -> Optional[str]:
        """Get env factory name for (robot_type, model_id)."""
        e = self.get(robot_type, model_id)
        return e.env_factory_name if e else None

    def get_model_path(self, robot_type: str, model_id: str) -> Optional[str]:
        """Resolve model path if resolver is set."""
        e = self.get(robot_type, model_id)
        if e and e.model_path_resolver:
            try:
                return e.model_path_resolver()
            except Exception:
                return None
        return None


def get_robot_registry() -> RobotRegistry:
    """Get singleton registry."""
    global _registry
    if _registry is None:
        _registry = RobotRegistry()
        _register_builtins(_registry)
    return _registry


def _get_ant_path() -> str:
    try:
        import importlib.resources
        with importlib.resources.path("gymnasium.envs.mujoco.assets", "ant.xml") as p:
            return str(Path(p).resolve())
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        import gymnasium
        p = Path(gymnasium.__file__).parent / "envs" / "mujoco" / "assets" / "ant.xml"
        if p.exists():
            return str(p)
    except (ImportError, AttributeError):
        pass
    raise FileNotFoundError("ant.xml not found. Install gymnasium: pip install gymnasium")


def _get_humanoid_path() -> str:
    try:
        import importlib.resources
        with importlib.resources.path("gymnasium.envs.mujoco.assets", "humanoid.xml") as p:
            return str(Path(p).resolve())
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        import gymnasium
        p = Path(gymnasium.__file__).parent / "envs" / "mujoco" / "assets" / "humanoid.xml"
        if p.exists():
            return str(p)
    except (ImportError, AttributeError):
        pass
    raise FileNotFoundError("humanoid.xml not found. Install gymnasium: pip install gymnasium")


def _get_go2_path(mjx: bool = False) -> Optional[str]:
    """Unitree Go2 MJCF from mujoco_menagerie. mjx=True prefers go2_mjx.xml (MJX-compatible)."""
    import os
    files = ("go2_mjx.xml", "go2.xml") if mjx else ("go2.xml", "go2_mjx.xml")
    # 1. mujoco_menagerie (pip install mujoco-menagerie or git clone)
    try:
        import mujoco_menagerie
        base = Path(mujoco_menagerie.__file__).parent
        for f in files:
            p = base / "unitree_go2" / f
            if p.exists():
                return str(p)
    except (ImportError, AttributeError):
        pass
    # 2. MUJOCO_MENAGERIE_PATH env
    menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if menagerie:
        for f in files:
            p = Path(menagerie) / "unitree_go2" / f
            if p.exists():
                return str(p)
    # 3. Project third_party
    proj = Path(__file__).resolve().parents[2]
    for d in (proj / "third_party" / "mujoco_menagerie", proj / "mujoco_menagerie"):
        for f in files:
            p = d / "unitree_go2" / f
            if p.exists():
                return str(p)
    return None


def _get_g1_path() -> Optional[str]:
    """Resolve the Unitree G1 scene MJCF.

    Delegates to :mod:`genedynamics.robots.g1.assets`, which is the single
    source of truth for G1 asset locations.
    """
    try:
        from genedynamics.robots.g1.assets import g1_scene_path

        return g1_scene_path()
    except Exception:
        return None


def _get_h1_path() -> Optional[str]:
    """Unitree H1 MJCF from dial-mpc (if available)."""
    try:
        import importlib.resources
        with importlib.resources.path("dial_mpc.models.unitree_h1", "mjx_scene_h1_walk.xml") as p:
            return str(Path(p).resolve())
    except (ImportError, ModuleNotFoundError):
        pass
    return None


def _register_builtins(reg: RobotRegistry) -> None:
    """Register built-in robot models."""
    from genedynamics.tasks.quadruped.spec import QuadrupedTaskSpec
    from genedynamics.tasks.humanoid.spec import HumanoidTaskSpec

    # Quadruped: ant (default), go2 (placeholder - env not yet implemented)
    reg.register(
        robot_type="quadruped",
        model_id="ant",
        env_factory_name="quadruped_flat_physics",
        nq=15,
        nv=14,
        act_dim=8,
        model_path_resolver=_get_ant_path,
        spec_class=QuadrupedTaskSpec,
        description="MuJoCo ant (gymnasium)",
    )
    reg.register(
        robot_type="quadruped",
        model_id="flat",
        env_factory_name="quadruped_flat_physics",
        nq=15,
        nv=14,
        act_dim=8,
        model_path_resolver=_get_ant_path,
        spec_class=QuadrupedTaskSpec,
        description="Alias for ant, flat terrain",
    )
    reg.register(
        robot_type="quadruped",
        model_id="rough",
        env_factory_name="quadruped_rough_physics",
        nq=15,
        nv=14,
        act_dim=8,
        model_path_resolver=_get_ant_path,
        spec_class=QuadrupedTaskSpec,
        description="Ant on rough terrain",
    )
    reg.register(
        robot_type="quadruped",
        model_id="push",
        env_factory_name="quadruped_push_physics",
        nq=15,
        nv=14,
        act_dim=8,
        model_path_resolver=_get_ant_path,
        spec_class=QuadrupedTaskSpec,
        description="Ant push task",
    )
    # Go2: Unitree Go2 quadruped (mujoco_menagerie)
    reg.register(
        robot_type="quadruped",
        model_id="go2",
        env_factory_name="quadruped_go2_physics",
        nq=19,
        nv=18,
        act_dim=12,
        model_path_resolver=_get_go2_path,
        spec_class=QuadrupedTaskSpec,
        description="Unitree Go2 (mujoco_menagerie)",
    )

    # Humanoid: humanoid (default), h1 (reserved)
    reg.register(
        robot_type="humanoid",
        model_id="humanoid",
        env_factory_name="humanoid_simplified_physics",
        nq=24,
        nv=23,
        act_dim=17,
        model_path_resolver=_get_humanoid_path,
        spec_class=HumanoidTaskSpec,
        description="MuJoCo humanoid (gymnasium)",
    )
    reg.register(
        robot_type="humanoid",
        model_id="h1",
        env_factory_name="humanoid_simplified_physics",  # Fallback until humanoid_h1_physics exists
        nq=34,
        nv=33,
        act_dim=19,
        model_path_resolver=_get_h1_path,
        spec_class=HumanoidTaskSpec,
        description="Unitree H1 (dial-mpc)",
    )
    # G1: Unitree G1 humanoid (mujoco_menagerie)
    reg.register(
        robot_type="humanoid",
        model_id="g1",
        env_factory_name="humanoid_g1_physics",
        nq=36,  # G1: base + joints
        nv=35,
        act_dim=23,
        model_path_resolver=_get_g1_path,
        spec_class=HumanoidTaskSpec,
        description="Unitree G1 (mujoco_menagerie)",
    )


def register_quadruped(
    model_id: str,
    env_factory_name: str,
    nq: int,
    nv: int,
    act_dim: int,
    model_path_resolver: Optional[Callable[[], str]] = None,
    description: str = "",
) -> None:
    """Convenience: register a quadruped model."""
    get_robot_registry().register(
        robot_type="quadruped",
        model_id=model_id,
        env_factory_name=env_factory_name,
        nq=nq,
        nv=nv,
        act_dim=act_dim,
        model_path_resolver=model_path_resolver,
        description=description,
    )


def register_humanoid(
    model_id: str,
    env_factory_name: str,
    nq: int,
    nv: int,
    act_dim: int,
    model_path_resolver: Optional[Callable[[], str]] = None,
    description: str = "",
) -> None:
    """Convenience: register a humanoid model."""
    get_robot_registry().register(
        robot_type="humanoid",
        model_id=model_id,
        env_factory_name=env_factory_name,
        nq=nq,
        nv=nv,
        act_dim=act_dim,
        model_path_resolver=model_path_resolver,
        description=description,
    )
