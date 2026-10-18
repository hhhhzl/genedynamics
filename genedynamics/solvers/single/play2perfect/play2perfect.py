"""Play2Perfect solver frontend and pinned published-checkpoint loader."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch


_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_LOCK = _ROOT / "requirements/baselines/sources.lock.json"


def _verified_source_checkout() -> Path:
    """Resolve the pinned source checkout used by this solver."""
    specs = json.loads(_SOURCE_LOCK.read_text())["sources"]
    spec = next(item for item in specs if item["method"] == "play2perfect")
    source = _ROOT / "baselines" / spec["directory"]
    if source.is_symlink() or not (source / ".git").exists():
        raise FileNotFoundError(f"pinned Play2Perfect source missing: {source}")

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source), *args],
            text=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        ).strip()

    if git("rev-parse", "HEAD") != spec["commit"]:
        raise ValueError("Play2Perfect checkout is not at its locked commit")
    if git("remote", "get-url", "origin") != spec["repository"]:
        raise ValueError("Play2Perfect checkout has an unexpected origin")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Play2Perfect checkout has tracked modifications")
    for name, digest in spec["licenses"].items():
        path = source / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Play2Perfect license mismatch: {name}")
    missing = [path for path in spec["required_source_files"] if not (source / path).is_file()]
    if missing:
        raise ValueError(f"Play2Perfect source files missing: {missing}")
    return source


def _numpy1_scalar(*args):
    core = getattr(np, "_core", np.core)
    return core.multiarray.scalar(*args)


_numpy1_scalar.__module__ = "numpy.core.multiarray"
_numpy1_scalar.__name__ = "scalar"
_numpy1_scalar.__qualname__ = "scalar"


def _published_payload(path: str | Path):
    allowed = [_numpy1_scalar, np.dtype, type(np.dtype("float64")), type(np.dtype("float32"))]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return payload[0] if 0 in payload else payload


class Play2PerfectSolver:
    """Thin solver around the original recurrent rl-games policy."""

    def __init__(self, model: Any, *, observation_dim: int, device: str = "cpu"):
        from .backends.play2perfect_torch import Play2PerfectBackendTorch

        self._backend_impl = Play2PerfectBackendTorch(
            model, observation_dim=observation_dim, device=device
        )

    @classmethod
    def from_published_checkpoint(cls, checkpoint: str | Path, *, device: str = "cpu", batch_size: int = 1):
        import yaml

        source = _verified_source_checkout()
        rl_games = str(source / "rl_games")
        if rl_games not in sys.path:
            sys.path.insert(0, rl_games)
        from rl_games.algos_torch.model_builder import ModelBuilder

        payload = _published_payload(checkpoint)
        state = payload["model"]
        observation_dim = int(state["running_mean_std.running_mean"].numel())
        action_dim = int(state["a2c_network.mu.bias"].numel())
        recipe = source / "isaacsimenvs/cfg/train/PreciseAssemblySAPG.yaml"
        params = yaml.safe_load(recipe.read_text())["params"]
        target = torch.device("cuda" if device == "gpu" else device)
        model = ModelBuilder().load(params).build({
            "actions_num": action_dim,
            "input_shape": (observation_dim + 1,),
            "num_seqs": int(batch_size),
            "value_size": 1,
            "normalize_value": True,
            "normalize_input": True,
            "type": "extra_param",
            "coef_ids": torch.linspace(50.0, 0.0, 6, device=target),
            "coef_id_idx": observation_dim,
            "param_size": 32,
        }).to(target).eval()
        model.load_state_dict(state, strict=True)
        instance = cls(model, observation_dim=observation_dim, device=str(target))
        instance._backend_impl.reset(batch_size)
        return instance

    def reset(self, batch_size: int | None = None) -> None:
        self._backend_impl.reset(batch_size)

    def act(self, observation: Any, *, coefficient: Any = 0.0):
        return self._backend_impl.act(observation, coefficient=coefficient)


__all__ = ["Play2PerfectSolver"]
