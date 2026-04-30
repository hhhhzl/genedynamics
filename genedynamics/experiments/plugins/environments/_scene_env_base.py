"""
Shared base for scene-reconstruction environment plugins.

A "3DGS env" doesn't have classical dynamics — its job is to hand the solver an
`ObservationBundle` loaded from a dataset adapter. The base class here captures:

  - Dataset-root path resolution (relative to project root).
  - Lazy dataset loading + caching.
  - `reset()` returning the first camera pose as a stand-in state.
  - A placeholder `step()` that just re-returns the initial pose.

Subclasses only need to provide:
  - `_make_adapter(config) -> SceneDatasetAdapter`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from genedynamics.data import ObservationBundle, SceneDatasetAdapter


class SceneEnvBase:
    """Base class for scene-dataset environments used by 3DGS experiments."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.horizon = config.get("horizon", 10)
        self.dt = config.get("dt", 0.1)
        self.target = np.zeros(3, dtype=np.float32)
        self._observations: Optional[ObservationBundle] = None
        self._adapter: Optional[SceneDatasetAdapter] = None

        resolved = self._resolve_dataset_root(config)
        if resolved is not None:
            # Replace the config's dataset_root with the absolute path so
            # subclasses don't need to repeat the resolution logic.
            config = {**config, "dataset_root": str(resolved)}
            self.config = config
        try:
            self._adapter = self._make_adapter(config)
        except FileNotFoundError:
            if not config.get("use_stub_when_unavailable", False):
                raise

    def _resolve_dataset_root(self, config: Dict[str, Any]) -> Optional[Path]:
        dataset_root = config.get("dataset_root") or config.get("dataset_path")
        if not dataset_root:
            return None
        root = Path(dataset_root).expanduser()
        if root.is_absolute():
            return root if root.exists() else None
        for base in [Path.cwd(), Path(__file__).resolve().parents[4]]:
            candidate = base / root
            if candidate.exists():
                return candidate
        return None

    def _make_adapter(self, config: Dict[str, Any]) -> SceneDatasetAdapter:
        raise NotImplementedError

    def _stub_bundle(self) -> ObservationBundle:
        cfg = self.config
        n = int(cfg.get("max_views", 11))
        h = int(cfg.get("resolution_infer", cfg.get("image_height", 128)))
        w = int(cfg.get("resolution_infer", cfg.get("image_width", 128)))
        rng = np.random.default_rng(cfg.get("shuffle_seed", 0))
        images = (rng.standard_normal((n, h, w, 3)).astype(np.float32) * 0.1 + 0.5)
        poses = rng.standard_normal((n, 7)).astype(np.float32) * 0.1
        poses[:, 3:7] /= np.linalg.norm(poses[:, 3:7], axis=1, keepdims=True) + 1e-8
        return ObservationBundle(
            images=images, camera_poses=poses, intrinsics=None,
            timestamps=np.arange(n, dtype=np.float32),
        )

    # ---- interface used by experiment runner / solver ----

    def reset(self, rng: Any = None):
        obs = self.get_observations()
        n = obs.num_views
        if n > 0:
            initial = np.asarray(obs.camera_poses[0], dtype=np.float32)
        else:
            initial = np.zeros(7, dtype=np.float32)
            initial[3] = 1.0
        return initial, {}

    def get_observations(self) -> ObservationBundle:
        if self._observations is not None:
            return self._observations
        if self._adapter is not None:
            self._observations = self._adapter.load_bundle(
                split=self.config.get("split", "train"),
            )
        else:
            self._observations = self._stub_bundle()
        return self._observations

    def step(self, action: Any):
        return self.reset()[0], 0.0, False, {}
