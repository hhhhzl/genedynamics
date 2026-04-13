"""
Typed plan schema objects shared by corridor followers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


DEFAULT_CORRIDOR_FIELDS: Dict[str, int] = {
    "x": 0,
    "y": 1,
    "psi": 2,
    "h": 3,
    "psi_torso": 4,
    "a_left": 5,
    "a_right": 6,
    "p_left": 7,
    "p_right": 8,
    "v_x": 9,
    "v_y": 10,
    "omega": 11,
    "h_dot": 12,
    "psi_dot_torso": 13,
}


@dataclass(frozen=True)
class CorridorPlanSchema:
    """
    Semantic layout for corridor plan states.

    The follower stack only depends on semantic field names. Moving from 14D to
    16D should only require updating this schema and the plan adapter.
    """

    version: str = "corridor_plan_v1"
    fields: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_CORRIDOR_FIELDS))
    state_dim: int = 14

    def __post_init__(self) -> None:
        field_to_idx = {str(k): int(v) for k, v in self.fields.items()}
        if len(set(field_to_idx.values())) != len(field_to_idx):
            raise ValueError("CorridorPlanSchema fields must map to unique indices")
        object.__setattr__(self, "fields", field_to_idx)

    @classmethod
    def corridor_14d(cls) -> "CorridorPlanSchema":
        return cls(version="corridor_plan_v1", fields=dict(DEFAULT_CORRIDOR_FIELDS), state_dim=14)

    def with_extra_fields(self, **field_to_index: int) -> "CorridorPlanSchema":
        fields = dict(self.fields)
        next_dim = int(self.state_dim)
        for name, idx in field_to_index.items():
            fields[str(name)] = int(idx)
            next_dim = max(next_dim, int(idx) + 1)
        return CorridorPlanSchema(version=self.version, fields=fields, state_dim=next_dim)

    def index(self, field_name: str) -> int:
        if field_name not in self.fields:
            raise KeyError(f"Field '{field_name}' not found in schema {self.version}")
        return int(self.fields[field_name])

    def has(self, field_name: str) -> bool:
        return field_name in self.fields

    def get_value(self, state: Sequence[float], field_name: str, default: float = 0.0) -> float:
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        idx = self.fields.get(field_name)
        if idx is None or int(idx) >= arr.size:
            return float(default)
        return float(arr[int(idx)])

    def decode_state(self, state: Sequence[float], *, time_sec: float = 0.0) -> "CorridorPlanFrame":
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        if arr.size < self.state_dim:
            raise ValueError(f"Expected state_dim >= {self.state_dim}, got {arr.size}")
        base_fields = set(DEFAULT_CORRIDOR_FIELDS)
        extras: Dict[str, float] = {}
        for name, idx in self.fields.items():
            if name in base_fields:
                continue
            if int(idx) < arr.size:
                extras[name] = float(arr[int(idx)])
        return CorridorPlanFrame(
            time_sec=float(time_sec),
            x=self.get_value(arr, "x"),
            y=self.get_value(arr, "y"),
            psi=self.get_value(arr, "psi"),
            h=self.get_value(arr, "h"),
            psi_torso=self.get_value(arr, "psi_torso"),
            a_left=self.get_value(arr, "a_left"),
            a_right=self.get_value(arr, "a_right"),
            p_left=self.get_value(arr, "p_left"),
            p_right=self.get_value(arr, "p_right"),
            v_x=self.get_value(arr, "v_x"),
            v_y=self.get_value(arr, "v_y"),
            omega=self.get_value(arr, "omega"),
            h_dot=self.get_value(arr, "h_dot"),
            psi_dot_torso=self.get_value(arr, "psi_dot_torso"),
            extras=extras,
            raw_state=arr.astype(np.float64, copy=True),
        )


@dataclass
class CorridorPlanFrame:
    time_sec: float
    x: float
    y: float
    psi: float
    h: float
    psi_torso: float
    a_left: float
    a_right: float
    p_left: float
    p_right: float
    v_x: float
    v_y: float
    omega: float
    h_dot: float
    psi_dot_torso: float
    extras: Dict[str, float] = field(default_factory=dict)
    raw_state: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64), repr=False)

    @property
    def xy(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float64)

    @property
    def planar_speed(self) -> float:
        return float(np.linalg.norm([self.v_x, self.v_y]))


@dataclass
class CorridorPlanTrajectory:
    states: np.ndarray
    times: np.ndarray
    schema: CorridorPlanSchema
    actions: Optional[np.ndarray] = None
    best_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        if self.states.ndim != 2:
            raise ValueError(f"Expected 2D states array, got shape {self.states.shape}")
        if self.states.shape[0] != self.times.shape[0]:
            raise ValueError("times and states must have matching leading dimension")
        if self.actions is not None:
            self.actions = np.asarray(self.actions, dtype=np.float64)

    @property
    def dt(self) -> float:
        if self.times.size <= 1:
            return 0.0
        return float(np.median(np.diff(self.times)))

    def decode_frames(self) -> List[CorridorPlanFrame]:
        return [
            self.schema.decode_state(self.states[i], time_sec=float(self.times[i]))
            for i in range(self.states.shape[0])
        ]
