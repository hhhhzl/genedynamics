"""MDAC solver-local helpers.

MDAC does NOT re-implement constraint / schedule / geometry / transport math —
those are REUSED from upstream packages (this was the whole point of the
2026-06-22 rework):

  soft-feasibility / iALM    -> the cfsmbd (mdcoas) AL pattern: augmented reward
                                (aug_lambda/aug_rho + [g]_+ penalty) via the env's
                                constraint_residual hook (build_brax_rollout_augmented).
                                NB: iALM is cfsmbd's, NOT mdoc's.
  action-space ConstraintFilter -> genedynamics/core/constraints/action_filters
                                (this is mdoc's mechanism — distinct from iALM)
  DDPM/DDIM/FM adaptive      -> genedynamics/solvers/common/transport  (AdaptiveTransport)
  metric + tangent proj + retraction -> genedynamics/genemetry  (SdfManifold, CfsRetraction)
  coupled annealing          -> genedynamics/genemetry/schedule (ScheduleOverlay)
                                + genedynamics/core/constraints/schedulers
  position-stiffness SPD primitive -> genedynamics/core/control/stiffness  (UPSTREAM)

The only solver-local module here is `method_registry` — the MDAC ablation table
(one flag per component) + the fairness self-check (same `(M,H,K)`), which is
MDAC-specific configuration, not reusable math.
"""
