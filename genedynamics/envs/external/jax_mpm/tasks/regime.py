"""RegimeSpec: a single train/test entry in the writeup §10.3 regime bank.

A regime captures everything that varies BETWEEN evaluations of the same
(morphology, controller) pair: terrain shape, ground friction, and (for the
push task) manipuland mass/friction. It is the unit indexed by `mode_id`
in the rollout API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..terrain import (
    TerrainSpec,
    flat_terrain,
    slope_terrain,
    gauss_bumps_terrain,
    soft_patch_terrain,
    ridge_terrain,
    gap_terrain,
    mixed_terrain,
)
from ..manipuland import ManipulandConfig


@dataclass(frozen=True)
class RegimeSpec:
    """A single regime: terrain + friction + (optional) manipuland.

    Fields
    ------
    name : str
        Unique identifier for serialization / metrics.
    terrain : TerrainSpec
    friction : float
        Ground friction coefficient μ_g.
    manipuland : Optional[ManipulandConfig]
        Push-task object descriptor. None for locomotion tasks.
    """

    name: str
    terrain: TerrainSpec
    friction: float
    manipuland: Optional[ManipulandConfig] = None

    @property
    def has_manipuland(self) -> bool:
        return self.manipuland is not None


# ---------------------------------------------------------------------------
# Banks (writeup §10.3): train uses easy regimes, test holds out hard ones.
# ---------------------------------------------------------------------------


def _terrain_set_train(n_grid: int) -> List[TerrainSpec]:
    return [
        flat_terrain(n_grid),
        slope_terrain(n_grid, angle_deg=5.0),
        gauss_bumps_terrain(n_grid, n_bumps=4, seed=1),
        soft_patch_terrain(n_grid, seed=2),
    ]


def _terrain_set_test(n_grid: int) -> List[TerrainSpec]:
    return [
        slope_terrain(n_grid, angle_deg=10.0),
        gauss_bumps_terrain(n_grid, n_bumps=8, seed=11),
        ridge_terrain(n_grid),
        gap_terrain(n_grid),
        mixed_terrain(n_grid, seed=12),
    ]


def make_train_bank(
    n_grid: int = 64,
    task: str = "locomotion",
    frictions=(0.4, 0.6, 0.8),
    masses=(0.5, 1.0, 1.5),
) -> List[RegimeSpec]:
    """Writeup §10.3 training bank.

    Args
    ----
    task : "locomotion" or "push"
    frictions : ground-friction values to cross with each terrain
    masses : object masses (push only)
    """
    bank: List[RegimeSpec] = []
    if task == "locomotion":
        for t in _terrain_set_train(n_grid):
            for mu in frictions:
                bank.append(RegimeSpec(
                    name=f"train__{t.name}__mu{mu:.2f}",
                    terrain=t, friction=float(mu),
                ))
    elif task == "push":
        for t in _terrain_set_train(n_grid):
            for mu in frictions:
                for M in masses:
                    bank.append(RegimeSpec(
                        name=f"train__{t.name}__mu{mu:.2f}__M{M:.2f}",
                        terrain=t, friction=float(mu),
                        manipuland=ManipulandConfig(mass=float(M)),
                    ))
    elif task == "carry":
        # Carry/transport: object falls under gravity (horizontal_only=False) and
        # must be supported by the body. Regimes cross terrain (incl. slopes) ×
        # friction × object mass — the writeup §carry mass/slope regime set.
        for t in _terrain_set_train(n_grid):
            for mu in frictions:
                for M in masses:
                    bank.append(RegimeSpec(
                        name=f"train__carry__{t.name}__mu{mu:.2f}__M{M:.2f}",
                        terrain=t, friction=float(mu),
                        manipuland=ManipulandConfig(mass=float(M), horizontal_only=False),
                    ))
    else:
        raise ValueError(f"Unknown task {task!r}")
    return bank


def make_test_bank(
    n_grid: int = 64,
    task: str = "locomotion",
    frictions=(0.3, 0.5, 0.7, 0.9),
    masses=(0.75, 1.25, 1.75),
) -> List[RegimeSpec]:
    """Held-out evaluation bank — different terrains, different frictions."""
    bank: List[RegimeSpec] = []
    if task == "locomotion":
        for t in _terrain_set_test(n_grid):
            for mu in frictions:
                bank.append(RegimeSpec(
                    name=f"test__{t.name}__mu{mu:.2f}",
                    terrain=t, friction=float(mu),
                ))
    elif task == "push":
        for t in _terrain_set_test(n_grid):
            for mu in frictions:
                for M in masses:
                    bank.append(RegimeSpec(
                        name=f"test__{t.name}__mu{mu:.2f}__M{M:.2f}",
                        terrain=t, friction=float(mu),
                        manipuland=ManipulandConfig(mass=float(M)),
                    ))
    elif task == "carry":
        for t in _terrain_set_test(n_grid):
            for mu in frictions:
                for M in masses:
                    bank.append(RegimeSpec(
                        name=f"test__carry__{t.name}__mu{mu:.2f}__M{M:.2f}",
                        terrain=t, friction=float(mu),
                        manipuland=ManipulandConfig(mass=float(M), horizontal_only=False),
                    ))
    else:
        raise ValueError(f"Unknown task {task!r}")
    return bank
