from __future__ import annotations

"""
Offline preprocessing helpers for SafeDiffuser checkpoint conversion.

This module is used by `genedynamics.solvers.single.safediffuser.tools.convert_checkpoint`
to compute normalization mins/maxs (LimitsNormalizer) from dataset artifacts:
  - `dataset_config.pkl` (pickled diffuser Config; we unpickle via lightweight shims)
  - offline dataset files under `dataset_data_dir` (e.g., `env_*.pkl`)

It is intentionally NOT part of runtime inference; inference only reads:
  - `safediffuser_planning.yaml`
  - `state_*.pt`
"""

import pickle
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class Limits:
    obs_mins: np.ndarray  # (O,)
    obs_maxs: np.ndarray  # (O,)
    act_mins: np.ndarray  # (A,)
    act_maxs: np.ndarray  # (A,)


def _load_diffuser_dataset_config_pickle(path: str | Path) -> Dict[str, Any]:
    """Load DPCC/SafeDiffuser `dataset_config.pkl` without upstream imports.

    DPCC/SafeDiffuser logs store a pickled `diffuser.utils.config.Config` object that points to
    `diffuser.datasets.sequence.SequenceDataset`. Since upstream `diffuser` isn't required here,
    we shim the minimal module/class names needed for unpickling and only read the `_dict` field.
    """

    path = Path(path).expanduser().resolve()

    inserted: Dict[str, types.ModuleType] = {}

    def _ensure_module(name: str) -> types.ModuleType:
        import sys

        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        sys.modules[name] = m
        inserted[name] = m
        return m

    # Minimal shims to satisfy pickle global references.
    mod_diffuser = _ensure_module("diffuser")
    mod_diffuser.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("diffuser.utils")
    mod_config = _ensure_module("diffuser.utils.config")

    class Config:  # noqa: D401 - tiny shim
        """Pickle-compatible Config shim with `_dict`."""

        def __init__(self, _class=None, **kwargs: Any):
            self._class = _class
            self._dict = dict(kwargs)

        def __call__(self, *args: Any, **kwargs: Any):
            if self._class is None:
                raise TypeError("Config has no _class")
            merged = dict(self._dict)
            merged.update(kwargs)
            return self._class(*args, **merged)

    mod_config.Config = Config  # type: ignore[attr-defined]

    _ensure_module("diffuser.datasets")
    mod_sequence = _ensure_module("diffuser.datasets.sequence")

    class SequenceDataset:  # noqa: D401 - dummy class for unpickling
        """Dummy SequenceDataset class for unpickling only."""

        pass

    mod_sequence.SequenceDataset = SequenceDataset  # type: ignore[attr-defined]

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        d = getattr(obj, "_dict", None)
        if not isinstance(d, dict):
            raise TypeError(f"Unexpected dataset_config type {type(obj)}; missing _dict")
        return dict(d)
    finally:
        # Best-effort cleanup of shim modules we inserted.
        import sys

        for name in list(inserted.keys()):
            sys.modules.pop(name, None)


def _infer_avoiding_d3il_limits_from_data_dir(data_dir: str | Path) -> Limits:
    """Compute mins/maxs from d3il avoiding offline data.

    Expected file format: pickled dicts like `env_*.pkl` containing:
      - d["robot"]["des_c_pos"] : (T, >=2) desired end-effector position
      - d["robot"]["c_pos"]     : (T, >=2) measured end-effector position

    We map:
      obs[t] = [x_des, y_des, x, y]
      act[t] = des_xy[t+1] - des_xy[t]  (dx, dy)
    """

    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset data dir not found: {data_dir}")

    files = sorted(data_dir.glob("env_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No env_*.pkl files found under: {data_dir}")

    obs_min = np.full((4,), np.inf, dtype=np.float32)
    obs_max = np.full((4,), -np.inf, dtype=np.float32)
    act_min = np.full((2,), np.inf, dtype=np.float32)
    act_max = np.full((2,), -np.inf, dtype=np.float32)

    for p in files:
        with open(p, "rb") as f:
            d = pickle.load(f)
        robot = d.get("robot", {})
        des = np.asarray(robot["des_c_pos"], dtype=np.float32)
        pos = np.asarray(robot["c_pos"], dtype=np.float32)
        if des.ndim != 2 or pos.ndim != 2 or des.shape[0] != pos.shape[0]:
            raise ValueError(f"Unexpected shapes in {p.name}: des={des.shape}, pos={pos.shape}")
        if des.shape[1] < 2 or pos.shape[1] < 2:
            raise ValueError(f"Need at least 2D xy in {p.name}: des={des.shape}, pos={pos.shape}")

        des_xy = des[:, :2]
        pos_xy = pos[:, :2]
        obs = np.concatenate([des_xy, pos_xy], axis=-1)  # (T,4)
        if obs.shape[0] >= 1:
            obs_min = np.minimum(obs_min, obs.min(axis=0))
            obs_max = np.maximum(obs_max, obs.max(axis=0))
        if obs.shape[0] >= 2:
            act = des_xy[1:] - des_xy[:-1]  # (T-1,2)
            act_min = np.minimum(act_min, act.min(axis=0))
            act_max = np.maximum(act_max, act.max(axis=0))

    if not np.isfinite(obs_min).all() or not np.isfinite(obs_max).all():
        raise ValueError("Could not compute finite obs mins/maxs from dataset")
    if not np.isfinite(act_min).all() or not np.isfinite(act_max).all():
        raise ValueError("Could not compute finite act mins/maxs from dataset")

    return Limits(obs_mins=obs_min, obs_maxs=obs_max, act_mins=act_min, act_maxs=act_max)


def load_limits_from_dataset_artifacts(
    *,
    dataset_config_pkl: str | Path,
    dataset_data_dir: str | Path,
) -> Limits:
    """Compute mins/maxs for LimitsNormalizer from config + offline data.

    This is the "Route A" path:
    - `dataset_config.pkl` provides env + normalizer type (usually LimitsNormalizer)
    - mins/maxs are computed by scanning the offline dataset files under `dataset_data_dir`
    """

    cfg = _load_diffuser_dataset_config_pickle(dataset_config_pkl)
    env = str(cfg.get("env", ""))
    normalizer = str(cfg.get("normalizer", ""))

    if normalizer and normalizer != "LimitsNormalizer":
        raise ValueError(f"Only LimitsNormalizer supported for dataset-based loading, got: {normalizer}")

    if "avoiding-d3il" in env or env == "avoiding-d3il":
        return _infer_avoiding_d3il_limits_from_data_dir(dataset_data_dir)

    raise NotImplementedError(f"Dataset-based normalizer rebuild not implemented for env='{env}'")
