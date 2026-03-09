"""
3DGS integration backends for MBD3D.

- GaussianSplattingCLIBackend: subprocess wrapper for official 3DGS repo
- SceneParamsAdapter: SceneParams <-> 3DGS ply format
"""

from .adapters import SceneParamsToPlyAdapter, ply_to_scene_params
from .gaussian_splatting_cli_backend import GaussianSplattingCLIBackend

__all__ = [
    "GaussianSplattingCLIBackend",
    "SceneParamsToPlyAdapter",
    "ply_to_scene_params",
]
