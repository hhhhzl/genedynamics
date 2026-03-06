# 2GO Phase-0 Spec Freeze

This document freezes the minimum engineering contract for the `2go` method.

## Scope

- Target: 2D single integrator first (`single_2d`)
- Backend: JAX only for Phase-1
- Runtime path: must support `plan()` and `plan_batch()` in solver backend

## Result Schema (minimum)

The planning result dict must contain at least:

- `states`
- `actions`
- `candidate_states`
- `candidate_actions`
- `candidate_costs`
- `best_idx`

2GO-specific diagnostic fields (Phase-1 frozen keys):

- `gamma_hist`
- `sigma_hist`
- `delta_hist`
- `cvar_hist`
- `qp_call_hist`
- `activeK_hist`

Notes:

- In Phase-1, diagnostics may be proxy-based (from existing rollout/scheduler traces).
- Existing keys from `cfsmbd_jax` must remain unchanged.

## Acceptance Criteria (Phase-1)

- Functional:
  - `python genedynamics/experiments/runner.py <2go_config>.yaml` runs end-to-end.
  - `--dry-run` passes.
  - `num_modes > 1` with `mode_strategy=multirun` works.
  - `plan_batch` fast path remains available.

- Quality:
  - No regression in result schema required by experiment framework.
  - Keep compute graph on JAX path; avoid host-side algorithm changes in hot loops.

## Performance Guardrails

- Reuse existing high-performance implementation from `cfsmbd_jax` in Phase-1.
- Keep additional 2GO diagnostics lightweight and postprocess-only.
- Do not add heavy clustering/QP logic into Python outer loop in Phase-1.

