"""Validity checks for SoftBodySpec — placeholder gate for Phase 0.

Currently mostly trivial sanity checks. Real connectivity / support / sim
stability checks land in Phase 3 once mesh-based robotization is producing
candidates that can actually fail (default_robotize from a clean AABB never
fails any of these).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .protocols import SoftBodySpec


@dataclass(frozen=True)
class ValidityReport:
    ok: bool
    failures: List[str]

    def __bool__(self) -> bool:
        return self.ok


def _check_nonempty(spec: SoftBodySpec) -> List[str]:
    fails: List[str] = []
    if spec.n_particles == 0:
        fails.append("no particles")
    if spec.n_actuators <= 0:
        fails.append("n_actuators <= 0")
    if spec.n_voxels <= 0:
        fails.append("n_voxels <= 0")
    return fails


def _check_id_ranges(spec: SoftBodySpec) -> List[str]:
    fails: List[str] = []
    aid = np.asarray(spec.actuator_id)
    if aid.min(initial=0) < -1 or aid.max(initial=-1) >= spec.n_actuators:
        fails.append(
            f"actuator_id out of range: [{aid.min(initial=0)}, "
            f"{aid.max(initial=-1)}] vs allowed [-1, {spec.n_actuators - 1}]"
        )
    vid = np.asarray(spec.voxel_id)
    if vid.min(initial=0) < 0 or vid.max(initial=-1) >= spec.n_voxels:
        fails.append(
            f"voxel_id out of range: [{vid.min(initial=0)}, "
            f"{vid.max(initial=-1)}] vs allowed [0, {spec.n_voxels - 1}]"
        )
    return fails


def _check_fiber_norm(spec: SoftBodySpec, atol: float = 1e-3) -> List[str]:
    fails: List[str] = []
    norms = np.linalg.norm(np.asarray(spec.fiber_dirs), axis=-1)
    if np.any(np.abs(norms - 1.0) > atol):
        fails.append(f"fiber_dirs not unit-norm (max |‖d‖-1|={float(np.max(np.abs(norms-1))):.3g})")
    return fails


def _check_active_coverage(spec: SoftBodySpec, min_active_frac: float = 0.05) -> List[str]:
    fails: List[str] = []
    aid = np.asarray(spec.actuator_id)
    active = float(np.mean(aid >= 0))
    if active < min_active_frac:
        fails.append(
            f"too few active particles ({active:.2%} < {min_active_frac:.2%})"
        )
    return fails


def check_spec(
    spec: SoftBodySpec,
    *,
    min_active_frac: float = 0.05,
) -> ValidityReport:
    """Run all Phase-0 sanity checks. Returns ValidityReport(ok, failures).

    Future (Phase 3): connectivity (single connected component), ground
    support (at least one particle near floor), short-rollout stability
    (zero-action rollout doesn't blow up).
    """
    fails: List[str] = []
    fails.extend(_check_nonempty(spec))
    if not fails:  # downstream checks assume the basic invariants hold
        fails.extend(_check_id_ranges(spec))
        fails.extend(_check_fiber_norm(spec))
        fails.extend(_check_active_coverage(spec, min_active_frac=min_active_frac))
    return ValidityReport(ok=not fails, failures=fails)
