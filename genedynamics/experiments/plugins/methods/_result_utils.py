"""
Utilities to keep method-plugin result payloads schema-compatible.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def normalize_result_from_trajectory(
    result_traj: Any,
    *,
    initial_state: Any = None,
    preserve_info: bool = True,
) -> Dict[str, Any]:
    """
    Convert a Trajectory-like output into a stable dict schema.

    This helper is intentionally conservative:
    - preserve original info keys by default
    - only force-populate states/actions for downstream metrics/viz
    - never mutate planning semantics (best_idx/candidate_* unchanged)
    """
    out: Dict[str, Any] = {}
    if preserve_info and hasattr(result_traj, "info") and isinstance(result_traj.info, dict):
        out = result_traj.info

    if hasattr(result_traj, "states") and result_traj.states is not None:
        out["states"] = np.stack(result_traj.states, axis=0)
        if "initial_state" not in out:
            out["initial_state"] = out["states"][0] if len(out["states"]) > 0 else initial_state
    if hasattr(result_traj, "actions") and result_traj.actions is not None:
        out["actions"] = np.stack(result_traj.actions, axis=0)

    if "initial_state" not in out:
        out["initial_state"] = initial_state
    return out

