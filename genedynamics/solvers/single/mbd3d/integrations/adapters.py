"""
Adapters for SceneParams <-> 3DGS ply format.

3DGS ply structure (per vertex):
- x, y, z (position)
- f_dc_0, f_dc_1, f_dc_2 (SH degree 0 = RGB * 0.28209479177387814)
- opacity (logit)
- scale_0, scale_1, scale_2 (log scale)
- rot_0, rot_1, rot_2, rot_3 (quaternion w,x,y,z)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import SceneParams

# SH degree 0 factor for RGB <-> f_dc
SH_DC_FACTOR = 0.28209479177387814


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


class SceneParamsToPlyAdapter:
    """
    Convert SceneParams to/from 3DGS ply format.

    Handles coordinate conventions and value ranges.
    """

    def __init__(
        self,
        scale_is_log: bool = False,
        opacity_is_logit: bool = True,
        quat_order: str = "wxyz",
    ):
        self.scale_is_log = scale_is_log
        self.opacity_is_logit = opacity_is_logit
        self.quat_order = quat_order

    def scene_params_to_ply_dict(self, params: SceneParams) -> dict:
        """
        Convert SceneParams to dict of arrays for ply export.

        Returns:
            Dict with keys: x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
        """
        means = np.asarray(params.means, dtype=np.float32)
        scales = np.asarray(params.scales, dtype=np.float32)
        quats = np.asarray(params.quats, dtype=np.float32)
        opacities = np.asarray(params.opacities, dtype=np.float32).ravel()
        colors = np.asarray(params.colors, dtype=np.float32)

        n = means.shape[0]
        if opacities.size != n:
            opacities = np.broadcast_to(opacities, (n,)).ravel()
        if colors.shape[0] != n:
            colors = np.broadcast_to(colors, (n, 3))

        if not self.scale_is_log:
            scales = np.log(np.maximum(scales, 1e-6))
        if not self.opacity_is_logit:
            opacities = _logit(_sigmoid(opacities))

        f_dc = colors * SH_DC_FACTOR

        if self.quat_order == "wxyz":
            rot = quats
        else:
            rot = np.roll(quats, -1, axis=-1)

        return {
            "x": means[:, 0],
            "y": means[:, 1],
            "z": means[:, 2],
            "f_dc_0": f_dc[:, 0],
            "f_dc_1": f_dc[:, 1],
            "f_dc_2": f_dc[:, 2],
            "opacity": opacities,
            "scale_0": scales[:, 0],
            "scale_1": scales[:, 1],
            "scale_2": scales[:, 2],
            "rot_0": rot[:, 0],
            "rot_1": rot[:, 1],
            "rot_2": rot[:, 2],
            "rot_3": rot[:, 3],
        }

    def write_ply(
        self,
        params: SceneParams,
        path: str | Path,
        binary: bool = True,
    ) -> None:
        """Write SceneParams to ply file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        d = self.scene_params_to_ply_dict(params)
        n = len(d["x"])

        if binary:
            with open(path, "wb") as f:
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    "element vertex {}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property float f_dc_0\n"
                    "property float f_dc_1\n"
                    "property float f_dc_2\n"
                    "property float opacity\n"
                    "property float scale_0\n"
                    "property float scale_1\n"
                    "property float scale_2\n"
                    "property float rot_0\n"
                    "property float rot_1\n"
                    "property float rot_2\n"
                    "property float rot_3\n"
                    "end_header\n"
                ).format(n)
                f.write(header.encode("ascii"))
                for i in range(n):
                    f.write(
                        struct.pack(
                            "fff fff f fff ffff",
                            d["x"][i], d["y"][i], d["z"][i],
                            d["f_dc_0"][i], d["f_dc_1"][i], d["f_dc_2"][i],
                            d["opacity"][i],
                            d["scale_0"][i], d["scale_1"][i], d["scale_2"][i],
                            d["rot_0"][i], d["rot_1"][i], d["rot_2"][i], d["rot_3"][i],
                        )
                    )
        else:
            with open(path, "w") as f:
                f.write(
                    "ply\n"
                    "format ascii 1.0\n"
                    f"element vertex {n}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                    "property float opacity\n"
                    "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
                    "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
                    "end_header\n"
                )
                for i in range(n):
                    f.write(
                        f"{d['x'][i]} {d['y'][i]} {d['z'][i]} "
                        f"{d['f_dc_0'][i]} {d['f_dc_1'][i]} {d['f_dc_2'][i]} "
                        f"{d['opacity'][i]} "
                        f"{d['scale_0'][i]} {d['scale_1'][i]} {d['scale_2'][i]} "
                        f"{d['rot_0'][i]} {d['rot_1'][i]} {d['rot_2'][i]} {d['rot_3'][i]}\n"
                    )


def ply_to_scene_params(
    ply_path: str | Path,
    scale_is_log: bool = True,
    opacity_is_logit: bool = True,
    quat_order: str = "wxyz",
) -> SceneParams:
    """
    Read ply file and convert to SceneParams.

    Assumes standard 3DGS ply format (no f_rest).
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"ply file not found: {ply_path}")

    with open(ply_path, "rb") as f:
        line = f.readline().decode("ascii").strip()
        if line != "ply":
            raise ValueError(f"Not a ply file: {ply_path}")
        n = 0
        fmt = None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("format"):
                fmt = line.split()[1]
            elif line == "end_header":
                break

        if fmt == "binary_little_endian":
            data = struct.unpack(f"{n * 16}f", f.read(n * 16 * 4))
            arr = np.array(data, dtype=np.float32).reshape(n, 16)
        else:
            arr = np.loadtxt(f, dtype=np.float32, max_rows=n)
            if arr.ndim == 1:
                arr = arr.reshape(n, -1)

    means = arr[:, :3]
    f_dc = arr[:, 3:6]
    opacity = arr[:, 6]
    scale = arr[:, 7:10]
    rot = arr[:, 10:14]

    colors = f_dc / SH_DC_FACTOR
    if not scale_is_log:
        scale = np.exp(scale)
    if not opacity_is_logit:
        opacity = _logit(_sigmoid(opacity))

    if quat_order == "xyzw":
        quats = np.roll(rot, 1, axis=-1)
    else:
        quats = rot

    return SceneParams(
        means=means,
        scales=scale,
        quats=quats,
        opacities=opacity.reshape(-1, 1),
        colors=colors,
        spherical_harmonics=None,
    )
