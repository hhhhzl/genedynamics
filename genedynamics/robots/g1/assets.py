"""Resolution of Unitree G1 MJCF assets.

Single source of truth for "where is the G1 model file on disk".

Two entry points:

* ``g1_mjcf_path()`` returns the bare robot model (``g1.xml``) for use cases
  that only need the kinematic tree (e.g. ``RobotModel`` introspection).
* ``g1_scene_path()`` returns a full scene XML (``scene.xml`` / ``scene_mjx.xml``)
  if available, falling back to ``g1.xml``. Use this for simulation rollouts
  that need ground plane, lighting and visual assets.

Resolution order (first hit wins):

1. Explicit ``override`` argument
2. ``MUJOCO_MENAGERIE_PATH`` env var (``$/unitree_g1/``)
3. Installed ``mujoco_menagerie`` package
4. Project tree: ``<repo_root>/third_party/mujoco_menagerie/unitree_g1/``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

__all__ = ["g1_mjcf_path", "g1_scene_path", "G1AssetNotFoundError"]


class G1AssetNotFoundError(FileNotFoundError):
    """Raised when no Unitree G1 MJCF can be located on disk."""


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MENAGERIE_SUBDIR = "unitree_g1"

_MJCF_FILES: tuple[str, ...] = ("g1.xml", "g1_mjx.xml")
_MJCF_FILES_MJX: tuple[str, ...] = ("g1_mjx.xml", "g1.xml")
_SCENE_FILES: tuple[str, ...] = ("scene.xml", "scene_mjx.xml")
_SCENE_FILES_MJX: tuple[str, ...] = ("scene_mjx.xml", "scene.xml")


def g1_mjcf_path(override: Optional[str | Path] = None, *, prefer_mjx: bool = False) -> str:
    """Return an absolute path to the G1 robot MJCF.

    Args:
        override: Optional explicit path. If supplied and exists, returned as-is.
        prefer_mjx: When ``True``, prefer the MJX-compatible variant
            (``g1_mjx.xml``). MJX requires CG / Newton solvers and disallows
            certain contact-pair geometries.

    Raises:
        G1AssetNotFoundError: if no candidate path exists.
    """
    files = _MJCF_FILES_MJX if prefer_mjx else _MJCF_FILES
    return _resolve(override, files)


def g1_scene_path(override: Optional[str | Path] = None, *, prefer_mjx: bool = False) -> str:
    """Return an absolute path to a G1 scene XML, falling back to the bare MJCF.

    Args:
        override: Optional explicit path. If it points to a ``g1.xml`` and a
            sibling ``scene.xml`` exists, the scene file is preferred.
        prefer_mjx: When ``True``, prefer the MJX-compatible scene
            (``scene_mjx.xml``). The default ``scene.xml`` ships richer
            collision geometry that MJX cannot handle.

    Raises:
        G1AssetNotFoundError: if no candidate path exists.
    """
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise G1AssetNotFoundError(path)
        return str(_prefer_sibling_scene(path, prefer_mjx=prefer_mjx))
    files = _SCENE_FILES_MJX + _MJCF_FILES_MJX if prefer_mjx else _SCENE_FILES + _MJCF_FILES
    return _resolve(None, files)


def _resolve(override: Optional[str | Path], filenames: Sequence[str]) -> str:
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise G1AssetNotFoundError(path)
        return str(path)

    for base in _candidate_bases():
        for name in filenames:
            candidate = base / name
            if candidate.exists():
                return str(candidate.resolve())

    raise G1AssetNotFoundError(
        "Unable to locate Unitree G1 MJCF. Set MUJOCO_MENAGERIE_PATH, "
        "install the mujoco_menagerie package, or place it under "
        f"{_REPO_ROOT / 'third_party' / 'mujoco_menagerie'}."
    )


def _candidate_bases() -> Iterable[Path]:
    env_root = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if env_root:
        yield Path(env_root).expanduser() / _MENAGERIE_SUBDIR

    try:
        import mujoco_menagerie  # type: ignore[import-not-found]

        yield Path(mujoco_menagerie.__file__).parent / _MENAGERIE_SUBDIR
    except Exception:
        pass

    yield _REPO_ROOT / "third_party" / "mujoco_menagerie" / _MENAGERIE_SUBDIR
    yield _REPO_ROOT / "mujoco_menagerie" / _MENAGERIE_SUBDIR


def _prefer_sibling_scene(path: Path, *, prefer_mjx: bool = False) -> Path:
    if path.name not in _MJCF_FILES:
        return path
    scene_files = _SCENE_FILES_MJX if prefer_mjx else _SCENE_FILES
    for scene_name in scene_files:
        scene_path = path.parent / scene_name
        if scene_path.exists():
            return scene_path
    return path
