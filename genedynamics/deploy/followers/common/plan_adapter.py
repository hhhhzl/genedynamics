"""
Load and resample corridor planning outputs into typed plan trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from genedynamics.deploy.followers.common.plan_schema import (
    CorridorPlanSchema,
    CorridorPlanTrajectory,
)


def load_corridor_plan_from_seed_dir(
    seed_dir: str | Path,
    *,
    schema: Optional[CorridorPlanSchema] = None,
    best_idx: Optional[int] = None,
    source_dt: float = 0.25,
    target_dt: float = 0.02,
) -> CorridorPlanTrajectory:
    adapter = CorridorTrajectoryAdapter(
        schema=schema,
        source_dt=source_dt,
        target_dt=target_dt,
    )
    return adapter.load_seed_dir(seed_dir, best_idx=best_idx)


class CorridorTrajectoryAdapter:
    """
    Bridge from `trajectory.json` candidate states to follower-ready plan frames.

    This is the single module that should change when the planner state grows
    from 14D to 16D or when a new schema version is introduced.
    """

    def __init__(
        self,
        *,
        schema: Optional[CorridorPlanSchema] = None,
        source_dt: float = 0.25,
        target_dt: float = 0.02,
    ) -> None:
        self.schema = schema or CorridorPlanSchema.corridor_14d()
        self.source_dt = float(source_dt)
        self.target_dt = float(target_dt)

    def load_seed_dir(
        self,
        seed_dir: str | Path,
        *,
        best_idx: Optional[int] = None,
    ) -> CorridorPlanTrajectory:
        seed_path = Path(seed_dir)
        traj_path = seed_path / "trajectory" / "trajectory.json"
        if not traj_path.exists():
            raise FileNotFoundError(traj_path)
        return self.load_trajectory_json(
            traj_path,
            best_idx=best_idx,
            metadata={"seed_dir": str(seed_path)},
        )

    def load_trajectory_json(
        self,
        traj_path: str | Path,
        *,
        best_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CorridorPlanTrajectory:
        path = Path(traj_path)
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f) or {}
        candidates = blob.get("candidate_states") or []
        if not candidates:
            raise ValueError(f"No candidate_states in {path}")
        idx = int(blob.get("best_idx", 0) if best_idx is None else best_idx)
        idx = max(0, min(idx, len(candidates) - 1))
        states = np.asarray(candidates[idx], dtype=np.float64)

        actions = None
        cand_actions = blob.get("candidate_actions") or []
        if idx < len(cand_actions):
            actions = np.asarray(cand_actions[idx], dtype=np.float64)

        base_meta: Dict[str, Any] = {
            "trajectory_json": str(path),
            "best_idx": idx,
            "source_dt": self.source_dt,
            "target_dt": self.target_dt,
        }
        if metadata:
            base_meta.update(metadata)

        times = np.arange(states.shape[0], dtype=np.float64) * float(self.source_dt)
        traj = CorridorPlanTrajectory(
            states=states,
            times=times,
            schema=self.schema,
            actions=actions,
            best_idx=idx,
            metadata=base_meta,
        )
        return self.resample(traj)

    def resample(self, traj: CorridorPlanTrajectory) -> CorridorPlanTrajectory:
        if traj.states.shape[0] <= 1:
            return traj
        if self.target_dt <= 0.0 or abs(self.target_dt - traj.dt) <= 1e-9:
            return traj

        old_t = traj.times
        new_t = np.arange(0.0, float(old_t[-1]) + 0.5 * self.target_dt, self.target_dt, dtype=np.float64)
        new_states = np.zeros((new_t.shape[0], traj.states.shape[1]), dtype=np.float64)

        unwrap_fields = {"psi", "psi_torso"}
        for dim in range(traj.states.shape[1]):
            values = traj.states[:, dim]
            if dim in self._schema_indices(unwrap_fields):
                values = np.unwrap(values)
            new_values = np.interp(new_t, old_t, values)
            new_states[:, dim] = new_values

        if traj.actions is not None and traj.actions.ndim == 2 and traj.actions.shape[0] == traj.states.shape[0]:
            new_actions = np.zeros((new_t.shape[0], traj.actions.shape[1]), dtype=np.float64)
            for dim in range(traj.actions.shape[1]):
                new_actions[:, dim] = np.interp(new_t, old_t, traj.actions[:, dim])
        else:
            new_actions = traj.actions

        meta = dict(traj.metadata)
        meta["resampled"] = True
        meta["source_num_frames"] = int(traj.states.shape[0])
        meta["target_num_frames"] = int(new_states.shape[0])
        return CorridorPlanTrajectory(
            states=new_states,
            times=new_t,
            schema=traj.schema,
            actions=new_actions,
            best_idx=traj.best_idx,
            metadata=meta,
        )

    def _schema_indices(self, names: set[str]) -> set[int]:
        out = set()
        for name in names:
            if self.schema.has(name):
                out.add(self.schema.index(name))
        return out
