"""
Canonical motion episode representation for robot motion visualization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _as_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _infer_model_id(state_dim: int) -> str:
    # G1 state is typically 71D (nq=36, nv=35), Go2 is 37D, Ant is 29D.
    if int(state_dim) >= 70:
        return "g1"
    return "go2" if int(state_dim) >= 35 else "ant"


@dataclass
class MotionEpisode:
    """
    Unified episode payload for motion replay rendering.
    """

    states: np.ndarray
    actions: Optional[np.ndarray] = None
    robot_type: str = "quadruped"
    model_id: str = "auto"
    fps: float = 20.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = _as_array(self.states)
        if self.actions is not None:
            self.actions = _as_array(self.actions)
        if self.model_id == "auto":
            self.model_id = _infer_model_id(self.states.shape[-1])

    @classmethod
    def from_deploy_episode_dir(cls, episode_dir: str | Path, fps: float = 20.0) -> "MotionEpisode":
        """
        Build from deploy replay directory containing states/actions npy files.
        """
        from genedynamics.execution.logging.episode_writer import EpisodeWriter

        ep_dir = Path(episode_dir)
        writer = EpisodeWriter(str(ep_dir.parent))
        data = writer.load_episode(ep_dir)
        states = np.asarray(data.get("states"), dtype=np.float64)
        actions = np.asarray(data.get("actions"), dtype=np.float64) if data.get("actions") is not None else None
        meta = dict(data.get("meta") or {})
        state_dim = int(states.shape[-1]) if states.ndim >= 2 else int(states.size)
        model_id = _infer_model_id(state_dim)
        return cls(
            states=states,
            actions=actions,
            robot_type=str(meta.get("robot_type", "quadruped")),
            model_id=str(meta.get("model_id", model_id)),
            fps=float(meta.get("fps", fps)),
            metadata=meta,
        )

    @classmethod
    def from_experiment_seed_dir(
        cls,
        seed_dir: str | Path,
        *,
        best_idx: Optional[int] = None,
        fps: float = 20.0,
    ) -> "MotionEpisode":
        """
        Build from experiments output directory:
        .../level_x/seed_y/{results.json, trajectory/trajectory.json}
        """
        seed_path = Path(seed_dir)
        traj_path = seed_path / "trajectory" / "trajectory.json"
        if not traj_path.exists():
            raise FileNotFoundError(f"trajectory.json not found: {traj_path}")

        with open(traj_path, "r", encoding="utf-8") as f:
            traj = json.load(f)

        candidates = traj.get("candidate_states") or []
        if not candidates:
            raise ValueError(f"No candidate_states in {traj_path}")

        idx = int(traj.get("best_idx", 0) if best_idx is None else best_idx)
        idx = max(0, min(idx, len(candidates) - 1))
        states = np.asarray(candidates[idx], dtype=np.float64)

        actions = None
        cand_actions = traj.get("candidate_actions") or []
        if idx < len(cand_actions):
            actions = np.asarray(cand_actions[idx], dtype=np.float64)

        result_meta: Dict[str, Any] = {}
        results_path = seed_path / "results.json"
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                result_meta = json.load(f) or {}

        state_dim = int(states.shape[-1]) if states.ndim >= 2 else int(states.size)
        model_id = _infer_model_id(state_dim)

        meta = {
            "source": "experiments",
            "seed_dir": str(seed_path),
            "best_idx": idx,
            "results": result_meta,
        }
        return cls(
            states=states,
            actions=actions,
            robot_type="quadruped",
            model_id=model_id,
            fps=fps,
            metadata=meta,
        )
