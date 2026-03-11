"""
SoftZoo vendor bootstrap and path resolution.

Ensures SoftZoo is importable and resolves canonical paths for:
- Code root (third_party/environments/softzoo)
- Config root (configs stay in third_party)
- Assets root (data/softzoo/assets for downloadable assets)

Environment variables (optional overrides):
- SOFTZOO_ROOT: Code root
- SOFTZOO_ASSETS_ROOT: Assets root (default: <project_root>/data/softzoo/assets)
- SOFTZOO_CONFIG_ROOT: Config root (default: <softzoo_root>/softzoo/configs)
"""

from __future__ import annotations

import os
import platform
import sys
import collections
import collections.abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


def _ensure_ossaudiodev_stub() -> None:
    """
    SoftZoo imports modules that reference Linux-only ossaudiodev.
    Provide a lightweight stub on non-Linux platforms.
    """
    if platform.system() == "Linux" or "ossaudiodev" in sys.modules:
        return

    import types

    stub = types.ModuleType("ossaudiodev")
    stub.SNDCTL_COPR_SENDMSG = 0
    stub.OSSAudioError = type("OSSAudioError", (Exception,), {})
    sys.modules["ossaudiodev"] = stub


def _ensure_collections_compat() -> None:
    """Compat shim for legacy attrdict imports on Python 3.10+."""
    for name in ("Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


@dataclass(frozen=True)
class SoftZooPaths:
    """
    Canonical paths for SoftZoo integration.

    Attributes:
        project_root: Project repository root
        code_root: third_party/environments/softzoo
        package_dir: softzoo Python package (code_root/softzoo)
        config_root: Config directory (package_dir/configs)
        env_configs_dir: Environment configs (config_root/env_configs)
        assets_root: Downloadable assets (data/softzoo/assets)
        meshes_pcd_dir: PCD meshes (assets_root/meshes/pcd)
        meshes_stl_dir: STL meshes (assets_root/meshes/stl or package)
        textures_dir: Textures (assets_root/textures or package)
    """

    project_root: Path
    code_root: Path
    package_dir: Path
    config_root: Path
    env_configs_dir: Path
    assets_root: Path
    meshes_pcd_dir: Path
    meshes_stl_dir: Path
    textures_dir: Path

    def resolve_pcd_path(self, name: str) -> Path:
        """Resolve PCD mesh path (e.g. Caterpillar.pcd)."""
        return self.meshes_pcd_dir / f"{name}.pcd"

    def resolve_stl_path(self, name: str) -> Path:
        """Resolve STL mesh path."""
        return self.meshes_stl_dir / f"{name}.stl"

    def env_config_path(self, config_name: str) -> Path:
        """Resolve env config YAML path."""
        base = config_name if config_name.endswith(".yaml") else f"{config_name}.yaml"
        return self.env_configs_dir / base


def _find_project_root() -> Path:
    """Find project root (containing pyproject.toml, setup.py, or .git)."""
    candidates = [
        Path(__file__).resolve().parents[4],  # genedynamics/envs/external/softzoo -> repo
        Path.cwd(),
    ]
    for p in candidates:
        p = Path(p).resolve()
        if (p / "pyproject.toml").exists() or (p / "setup.py").exists():
            return p
        if (p / ".git").exists():
            return p
    return Path.cwd().resolve()


def _resolve_assets_root(project_root: Path) -> Path:
    """Resolve assets root: env var > default data/softzoo/assets."""
    env_val = os.environ.get("SOFTZOO_ASSETS_ROOT")
    if env_val:
        p = Path(env_val).resolve()
        if p.is_absolute():
            return p
        return (project_root / env_val).resolve()
    return (project_root / "data" / "softzoo" / "assets").resolve()


def _resolve_code_root(project_root: Path) -> Path:
    """Resolve SoftZoo code root."""
    env_val = os.environ.get("SOFTZOO_ROOT")
    if env_val:
        p = Path(env_val).resolve()
        if p.is_absolute():
            return p
        return (project_root / env_val).resolve()
    return (project_root / "third_party" / "environments" / "softzoo").resolve()


def get_softzoo_paths(project_root: Optional[Path] = None) -> SoftZooPaths:
    """
    Resolve and return canonical SoftZoo paths.

    Args:
        project_root: Optional project root. Inferred if not provided.

    Returns:
        SoftZooPaths with all resolved paths.
    """
    if project_root is None:
        project_root = _find_project_root()
    else:
        project_root = Path(project_root).resolve()

    code_root = _resolve_code_root(project_root)
    package_dir = code_root / "softzoo"
    config_root = package_dir / "configs"
    env_configs_dir = config_root / "env_configs"
    assets_root = _resolve_assets_root(project_root)
    pkg_assets = package_dir / "assets"

    # Prefer data/softzoo/assets; fallback to package assets if data not populated
    if assets_root.exists():
        meshes_pcd_dir = assets_root / "meshes" / "pcd"
        meshes_stl_dir = assets_root / "meshes" / "stl"
        textures_dir = assets_root / "textures"
    elif pkg_assets.exists():
        assets_root = pkg_assets
        meshes_pcd_dir = pkg_assets / "meshes" / "pcd"
        meshes_stl_dir = pkg_assets / "meshes" / "stl"
        textures_dir = pkg_assets / "textures"
    else:
        meshes_pcd_dir = assets_root / "meshes" / "pcd"
        meshes_stl_dir = assets_root / "meshes" / "stl"
        textures_dir = assets_root / "textures"

    return SoftZooPaths(
        project_root=project_root,
        code_root=code_root,
        package_dir=package_dir,
        config_root=config_root,
        env_configs_dir=env_configs_dir,
        assets_root=assets_root,
        meshes_pcd_dir=meshes_pcd_dir,
        meshes_stl_dir=meshes_stl_dir,
        textures_dir=textures_dir,
    )


def ensure_softzoo_on_path(project_root: Optional[Path] = None) -> Path:
    """
    Ensure SoftZoo is importable. Adds code_root to sys.path.

    SoftZoo is expected at:
      <project_root>/third_party/environments/softzoo/

    Returns:
        code_root Path.
    """
    paths = get_softzoo_paths(project_root)
    code_root = paths.code_root

    if not code_root.exists():
        raise FileNotFoundError(
            f"SoftZoo code root not found: {code_root}\n"
            "Run: git clone https://github.com/zswang666/softzoo.git "
            "third_party/environments/softzoo"
        )

    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    _ensure_collections_compat()
    _ensure_ossaudiodev_stub()

    return code_root


def validate_softzoo_environment(
    project_root: Optional[Path] = None,
    *,
    require_assets: bool = False,
    require_pcd: Optional[str] = None,
) -> Tuple[bool, list[str]]:
    """
    Validate SoftZoo environment. Returns (ok, list of error/warning messages).

    Args:
        project_root: Optional project root.
        require_assets: If True, assets_root must exist.
        require_pcd: If set (e.g. "Caterpillar"), PCD file must exist.

    Returns:
        (success, messages)
    """
    messages: list[str] = []
    ok = True

    try:
        paths = get_softzoo_paths(project_root)
    except Exception as e:
        return False, [f"Path resolution failed: {e}"]

    if not paths.code_root.exists():
        messages.append(f"Code root missing: {paths.code_root}")
        ok = False
    else:
        messages.append(f"Code root OK: {paths.code_root}")

    if not paths.package_dir.exists():
        messages.append(f"Package dir missing: {paths.package_dir}")
        ok = False

    if not paths.env_configs_dir.exists():
        messages.append(f"Env configs missing: {paths.env_configs_dir}")
        ok = False
    else:
        messages.append(f"Config root OK: {paths.env_configs_dir}")

    if require_assets and not paths.assets_root.exists():
        messages.append(
            f"Assets root missing: {paths.assets_root}. "
            "Create data/softzoo/assets and download assets. See scripts/third_party/setup_softzoo_assets.sh"
        )
        ok = False
    elif paths.assets_root.exists():
        messages.append(f"Assets root OK: {paths.assets_root}")
    else:
        messages.append(
            f"Assets root not found (optional): {paths.assets_root}. "
            "Using package assets if available."
        )

    if require_pcd:
        pcd_path = paths.resolve_pcd_path(require_pcd)
        if not pcd_path.exists():
            messages.append(
                f"PCD file missing: {pcd_path}. "
                f"Download from https://drive.google.com/drive/folders/1AYeZsr2ZMb1DkeOndQM0nBlNfx7dorUL"
            )
            ok = False
        else:
            messages.append(f"PCD OK: {pcd_path}")

    return ok, messages
