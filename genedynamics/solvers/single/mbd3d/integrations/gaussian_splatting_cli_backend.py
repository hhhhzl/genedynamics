"""
Gaussian Splatting CLI backend for 3DGS-MAP baseline.

Wraps the official 3D Gaussian Splatting repo (graphdeco-inria/gaussian-splatting)
via subprocess. Used for:
- Training (3DGS-MAP baseline)
- Rendering (given SceneParams or ply)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..types import SceneParams
from .adapters import SceneParamsToPlyAdapter


class GaussianSplattingCLIBackend:
    """
    Subprocess wrapper for official 3DGS repo.

    Requires:
    - 3DGS repo cloned and GAUSSIAN_SPLATTING_PATH set (or gs_path in constructor)
    - NeRF Synthetic data in standard layout (transforms_*.json)
    """

    def __init__(
        self,
        gs_path: Optional[str] = None,
        python_exe: Optional[str] = None,
        source_path: Optional[str] = None,
    ):
        self.gs_path = Path(gs_path or self._find_gs_path()).resolve()
        self.python_exe = python_exe or sys.executable
        self.source_path = Path(source_path) if source_path else self.gs_path
        self._adapter = SceneParamsToPlyAdapter(scale_is_log=False, opacity_is_logit=True)

    def _find_gs_path(self) -> str:
        import os
        p = os.environ.get("GAUSSIAN_SPLATTING_PATH")
        if p:
            return p
        candidates = [
            Path.cwd() / "gaussian-splatting",
            Path.cwd() / "3dgs",
            Path(__file__).resolve().parents[4] / "gaussian-splatting",
        ]
        for c in candidates:
            if (c / "train.py").exists():
                return str(c)
        raise FileNotFoundError(
            "3DGS repo not found. Set GAUSSIAN_SPLATTING_PATH or clone "
            "https://github.com/graphdeco-inria/gaussian-splatting"
        )

    def train(
        self,
        source_path: str | Path,
        output_path: str | Path,
        images: str = "images",
        eval: bool = True,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run 3DGS training (train.py).

        Args:
            source_path: Path to NeRF Synthetic object dir (e.g. data/nerf_synthetic/lego)
            output_path: Output directory for checkpoints
            images: Subdir name ("images" for standard NeRF layout)
            eval: Run evaluation after training
            extra_args: Additional args for train.py

        Returns:
            Dict with returncode, stdout, stderr
        """
        source_path = Path(source_path).resolve()
        output_path = Path(output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_exe,
            str(self.gs_path / "train.py"),
            "-s", str(source_path),
            "-m", str(output_path),
            "--images", images,
        ]
        if eval:
            cmd.append("-e")
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.gs_path),
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def render(
        self,
        model_path: str | Path,
        output_path: str | Path,
        iteration: int = -1,
        skip_train: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run 3DGS rendering (render.py).

        Args:
            model_path: Path to trained model (output of train)
            output_path: Directory for rendered images
            iteration: Checkpoint iteration (-1 = latest)
            skip_train: If True, skip re-running train (use existing)
            extra_args: Additional args for render.py

        Returns:
            Dict with returncode, stdout, stderr
        """
        model_path = Path(model_path).resolve()
        output_path = Path(output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        render_script = self.gs_path / "render.py"
        if not render_script.exists():
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "render.py not found in 3DGS repo",
            }

        cmd = [
            self.python_exe,
            str(render_script),
            "-m", str(model_path),
            "--output_path", str(output_path),
        ]
        if iteration >= 0:
            cmd.extend(["--iteration", str(iteration)])
        if skip_train:
            cmd.append("--skip_train")
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.gs_path),
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def export_scene_params_to_ply(
        self,
        params: SceneParams,
        ply_path: str | Path,
    ) -> Path:
        """Export SceneParams to ply for external rendering."""
        ply_path = Path(ply_path).resolve()
        self._adapter.write_ply(params, ply_path, binary=True)
        return ply_path

    def convert_nerf_synthetic_to_3dgs_input(
        self,
        dataset_root: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Convert NeRF Synthetic layout to 3DGS expected layout.

        3DGS expects: source_path/images/ (with 0.png, 1.png, ...) and
        cameras.json or similar. NeRF Synthetic has train/r_*.png and
        transforms_train.json.

        This creates a symlink/copy structure. For standard NeRF Synthetic,
        the 3DGS repo's scripts/convert.py can be used instead.
        """
        dataset_root = Path(dataset_root).resolve()
        output_path = Path(output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)

        with open(dataset_root / "transforms_train.json", "r") as f:
            meta = json.load(f)

        frames = meta.get("frames", [])
        for i, frame in enumerate(frames):
            fp = frame.get("file_path", "")
            if not fp:
                continue
            src = dataset_root / fp.replace("\\", "/").lstrip("./")
            for ext in ("", ".png", ".jpg"):
                p = src if ext else src.with_suffix(".png")
                if not p.suffix:
                    p = p.with_suffix(".png")
                if p.exists():
                    dst = images_dir / f"{i:04d}.png"
                    if not dst.exists():
                        import shutil
                        shutil.copy2(p, dst)
                    break

        return output_path
