"""Method/ablation flag registry + fairness self-check (MDAC).

The MDAC paper reports an ablation per bypassed component (`idea.txt` 993-1011).
Each ablation toggles exactly ONE boolean in :class:`MethodFlags`; everything
else (sample budget `(M,H,K)`, horizon, env, energy) is held fixed. This module
is the single source of truth that makes that guarantee mechanical:
``resolve_method`` builds the flags for a named method and ``assert_fair`` checks
that two methods share the same sample budget so a comparison is honest.

Pure Python, no JAX — import-safe everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields
from typing import Dict, Tuple


@dataclass(frozen=True)
class MethodFlags:
    """One switch per MDAC component (default = full MDAC).

    Setting a flag to its *off* value reproduces a baseline / ablation. With ALL
    flags off MDAC reduces byte-identically to DIAL (the regression gate).
    """

    use_stiffness: bool = True          # position-stiffness primitive
    log_spd_stiffness: bool = True      # log-Euclidean K=exp(S) (False = Euclidean/clamped)
    use_rl_prior: bool = True           # model-free prior p_psi
    use_mb_rollout: bool = True         # model-based closed-loop Boltzmann
    use_soft_feasibility: bool = True   # augmented-Lagrangian weighting
    use_tangent_projection: bool = True # metric tangent projection P_M^G
    use_retraction: bool = True         # linearized retraction QP
    use_force_manifold: bool = True     # include F_cmd=F_target in CLEAN manifold
    use_adaptive_schedule: bool = True  # coupled (sigma,rho,kappa,lam_psi) anneal
    # CPU MGA geometry program. Legacy MDAC keeps these off so previous results
    # remain reproducible; staged methods opt in explicitly.
    use_horizon_geometry: bool = False  # cumulative dense controls + time-indexed target
    use_geometry_gate: bool = False     # blend raw/projected + raw/retracted updates
    component_geometry_gate: bool = False  # per control block instead of one scalar
    use_realization_compensation: bool = False  # frozen real-EE bias in clean geometry
    use_controllability_geometry: bool = False  # finite-difference true response lift

    def off(self) -> "MethodFlags":
        """All-off => DIAL-equivalent weighted-mean (the degeneration)."""
        return MethodFlags(*([False] * len(fields(self))))


# Named methods -> the single flag(s) they flip relative to full MDAC.
# An ablation flips exactly one; baselines flip the minimal consistent set.
_FULL = MethodFlags()
# fair sampling baseline: shares the position-stiffness primitive + MB rollout,
# only the MDAC manifold / prior / anneal seams removed.
_BASELINE = replace(_FULL, use_rl_prior=False, use_soft_feasibility=False,
                    use_tangent_projection=False, use_retraction=False,
                    use_adaptive_schedule=False)
METHOD_TABLE: Dict[str, MethodFlags] = {
    "mdac": _FULL,
    # --- single-flag ablations (idea.txt 993-1011) ---
    "mdac_no_stiffness": replace(_FULL, use_stiffness=False),
    "mdac_fixed_stiffness": replace(_FULL, log_spd_stiffness=False),
    "mdac_euclid_stiffness": replace(_FULL, log_spd_stiffness=False),
    "mdac_no_rl_prior": replace(_FULL, use_rl_prior=False),
    "mdac_no_mb_rollout": replace(_FULL, use_mb_rollout=False),
    "mdac_no_softfeas": replace(_FULL, use_soft_feasibility=False),
    "mdac_no_tangent": replace(_FULL, use_tangent_projection=False),
    "mdac_no_retraction": replace(_FULL, use_retraction=False),
    "mdac_position_only": replace(_FULL, use_force_manifold=False),
    # Diagnostic aliases keep optimizer flags fixed while env reward terms change
    # through the corresponding experiment configs.
    "mdac_risk": _FULL,
    "mdac_position_only_risk": replace(_FULL, use_force_manifold=False),
    # Composite diagnostic: keep AL/schedule active while disabling both geometry
    # mechanisms. Unlike DIAL, this isolates geometry from the other MDAC seams.
    "mdac_no_geometry": replace(
        _FULL, use_tangent_projection=False, use_retraction=False
    ),
    "mdac_no_anneal": replace(_FULL, use_adaptive_schedule=False),
    # --- MGA CPU staged methods -------------------------------------------------
    # These are composite candidate methods, not one-flag ablations. Each stage
    # adds exactly the named mechanism on top of the preceding one.
    "mdac_horizon": replace(_FULL, use_horizon_geometry=True),
    "mdac_scalar_gate": replace(
        _FULL, use_horizon_geometry=True, use_geometry_gate=True
    ),
    "mdac_component_gate": replace(
        _FULL,
        use_horizon_geometry=True,
        use_geometry_gate=True,
        component_geometry_gate=True,
    ),
    # Same algorithm flags as mdac_component_gate; YAML owns the realized-risk
    # weights so Pareto points do not proliferate solver implementations.
    "mdac_component_gate_risk": replace(
        _FULL,
        use_horizon_geometry=True,
        use_geometry_gate=True,
        component_geometry_gate=True,
    ),
    "mdac_realization": replace(
        _FULL,
        use_horizon_geometry=True,
        use_realization_compensation=True,
    ),
    "mdac_realization_gate": replace(
        _FULL,
        use_horizon_geometry=True,
        use_geometry_gate=True,
        component_geometry_gate=True,
        use_realization_compensation=True,
    ),
    "mdac_controllable": replace(
        _FULL,
        use_horizon_geometry=True,
        use_controllability_geometry=True,
    ),
    "mdac_controllable_gate": replace(
        _FULL,
        use_horizon_geometry=True,
        use_geometry_gate=True,
        component_geometry_gate=True,
        use_controllability_geometry=True,
    ),
    # --- baselines in the same sampler. idea.txt §Baselines: they "optimize over
    # the same control sequence U" -> they SHARE the position-stiffness primitive
    # (use_stiffness on); only the MDAC manifold/prior/anneal seams are removed.
    # The solver-level seams (softfeas/tangent/retraction/prior/anneal) are all
    # off, so the reverse-diffusion is still DIAL-equivalent (byte-identity gate
    # is solver-level, independent of the env stiffness chart).
    "dial": _BASELINE,
    "mbd": _BASELINE,
    # "mppi" as an MDAC flag-degeneration is the same _BASELINE as dial; it is renamed
    # "dial_anchor" so the name "mppi" frees up for the standalone brax MPPI baseline
    # (a NEW registered solver dispatched by make_controller, not an MDAC flag-set).
    "dial_anchor": _BASELINE,
    "dial_nostiff": _FULL.off(),       # true all-off anchor (no shared primitive)
}


def resolve_method(name: str) -> MethodFlags:
    """Flags for a named method/ablation. Unknown name => full MDAC + warn-free
    default is intentionally an ERROR (fairness depends on an explicit table)."""
    if name not in METHOD_TABLE:
        raise KeyError(
            f"method '{name}' not in METHOD_TABLE (have: {sorted(METHOD_TABLE)})"
        )
    return METHOD_TABLE[name]


def diff_flags(a: MethodFlags, b: MethodFlags) -> Tuple[str, ...]:
    """Names of flags that differ between two methods (ablation audit)."""
    return tuple(
        f.name for f in fields(a) if getattr(a, f.name) != getattr(b, f.name)
    )


def assert_single_flag_ablation(base: str, ablation: str) -> None:
    """Guarantee an ablation differs from its base by exactly one flag."""
    d = diff_flags(resolve_method(base), resolve_method(ablation))
    if len(d) != 1:
        raise AssertionError(
            f"ablation '{ablation}' vs '{base}' differs in {len(d)} flags {d}; "
            "an ablation must toggle exactly one component."
        )


def assert_fair(budget_a: Tuple[int, int, int], budget_b: Tuple[int, int, int]) -> None:
    """Fairness self-check: two methods must share the sample budget (M,H,K) =
    (Nsample, horizon, Ndiffuse). Raises if not."""
    if tuple(budget_a) != tuple(budget_b):
        raise AssertionError(
            f"unfair comparison: budget {budget_a} != {budget_b} (M,H,K must match)."
        )


__all__ = [
    "MethodFlags",
    "METHOD_TABLE",
    "resolve_method",
    "diff_flags",
    "assert_single_flag_ablation",
    "assert_fair",
]
