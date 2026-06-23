"""MRMFMBD **ablation/baseline** package — the FROZEN legacy engine.

This is the isolated home (Stage 1 of the theory-faithful refactor in
``docs/REFACTOR_PLAN.md``) for the OLD mechanisms that are being removed from the
method path: the fixed coarse->fine fidelity ladder, the control-variate
multi-fidelity estimator, the post-hoc top-K SHAC refinement, the legacy
``reward``/``cvar`` regime-marginalization modes, mean-return certification, and
the inert ALM dual scaffold.

It exists ONLY to reproduce the pre-refactor numbers for the paper's Q2
(mode-blind) and Q3 (fixed-fidelity / control-variate) ablations, byte-for-byte.
The real method lives in ``genedynamics.solvers.single.mrmfmbd``; do NOT add
new-method logic here.

The fidelity_system / estimator_system subpackages and the backend are verbatim
copies of the pre-refactor ``mrmfmbd`` code, re-homed so the method package can be
gutted without touching reproducibility.
"""

from __future__ import annotations

from .backends import MRMFMBDLegacyBackend, LegacyMBDConfig
from .codesign import MRMFMBDAblationBaseline

__all__ = [
    "MRMFMBDLegacyBackend",
    "LegacyMBDConfig",
    "MRMFMBDAblationBaseline",
]
