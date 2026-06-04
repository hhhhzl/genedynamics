# Transport Unification Plan — DDPM / DDIM / FM (zero-risk)

Status: **plan** (not yet implemented)
Owner: solver / sampler layer
Scope: `genedynamics/solvers/single/{mbd,ebmbd,mdoc,mdcoas,cfsmbd,twogo}`

---

## 0. TL;DR

All current model-based solvers run the **same** deterministic reverse map: a
DDPM score-form update with the stochastic term off (`ρ = DDPM`, `σ_eff = 0`).
We want the reverse transport to be a **swappable, adaptively-schedulable
component** — DDPM / DDIM / FM today, Schrödinger-Bridge later — without risking
any of the already-working algorithms.

The mechanism is the one this codebase already uses for `noise_sampler`:
**None-default injection**. Add an optional `transport` component; when it is
`None` (the default) the solver runs the *verbatim* existing code path, so the
traced JAX graph — and therefore the numerical output — is **byte-identical**.
A **regression test** locks this in: every solver, default config, must match a
saved baseline exactly.

---

## 1. Goal & non-goals

**Goals**
- One reverse-transport interface shared by all MBD-family solvers.
- DDPM / DDIM / FM selectable (and per-step schedulable, `ρ_k`) via config.
- A clean seam for a future single-pass Schrödinger-Bridge transport.
- **Zero behavioral change** for every existing config/result.

**Non-goals**
- Changing the geometry (`G`, `P`), the CFS retraction, the soft prior, or the
  adaptive constraint scheduler. Those are orthogonal and untouched.
- Full IPF/Sinkhorn Schrödinger Bridge (breaks the single-`scan` structure —
  out of scope; only the single-pass SB transport is in scope later).

---

## 2. Current state (verified)

**The reverse map is DDPM score-form with `σ_eff = 0`.** Canonical instance,
`genedynamics/solvers/single/mbd/backends/mbd_jax.py:269-279`:

```
score      = (-Yi + sqrt(abar)*Ybar_weighted) / (1 - abar)     # MBD score  Ŝ
Yim1       = (Yi + (1 - abar)*score) / sqrt(alpha)             # DDPM score-form, NO sigma*z
Ybar_next  = Yim1 / sqrt(abar[idx-1])                          # renormalise
Ybar_next  = Ybar_next + extra_sigma * noise_extra             # small, decaying diversity kick
```

There is **no** DDPM `σ_k z_k` stochastic term; only a separate annealed
`extra_sigma` diversity term. EB-MBD / MDOC / MDCOAS / CFSMBD / 2GO each carry
their own copy of this skeleton (plus their solver-specific extras: energy
barrier, CFS filter, geometry). In the paper's unified notation
(`latex_2go/sections/method.tex`, `eq:unified_reverse_update_k`) every one of
them is the special case `ρ_k = DDPM`, `σ_eff = 0`.

**The injection precedent already exists.** Every backend routes unit noise
through an optional, None-defaulted `noise_sampler`:
- `mbd_jax.py:106-108`, `cfsmbd_jax.py:284-286`, `ebmbd_jax.py:197-199`
```
def _draw_unit_noise(self, shape, key, state):
    if self.noise_sampler is not None:
        return self.noise_sampler.sample(shape, key=key, state=state)
    return <default draw>          # <- unchanged path when None
```
We mirror this exactly for `transport`.

**A diffusion-schedule abstraction already exists.**
`genedynamics/core/constraints/schedulers/DiffusionScheduler/` (base + `fixed`
+ `dualcontrol`) already emits per-step `M_k, T_k, beta0/betaT/Ndiffuse` and is
configured under `diffusion_schedulers:` in the YAML. It is the natural place
to also emit the transport family `ρ_k` and its coefficients. `dualcontrol` is
the precedent for an *adaptive* diffusion scheduler.

---

## 3. Design

### 3.1 The transport operator `T^ρ`

Per `eq:unified_reverse_update_k`, the reverse step is

```
  τ̃_{k-1} = T_k^{ρ_k}( τ_k, τ̂_{1|k}, ε̂_k, Ŝ_{G,k} ) + σ_eff^{ρ_k} · P(τ_k) ξ_k
```

with the family-shared quantities

```
  τ̂_{1|k} = ( τ_k + (1-abar_k) Ŝ_k ) / sqrt(abar_k)      # predicted clean  (eq:ddim_clean_estimate_2go)
  ε̂_k     = ( τ_k - sqrt(abar_k) τ̂_{1|k} ) / sqrt(1-abar_k)   # residual    (eq:ddim_residual_direction_2go)
```

`τ̂_{1|k}` is the reward-weighted clean estimate the solvers already compute
(`Ybar_weighted`). **`ε̂_k` is not currently materialised** — adding it is one
line and is needed only by DDIM/FM.

Interface (numpy + jax backends, registry-friendly like genemetry):

```python
class ReverseTransport(ABC):
    def step(self, tau_k, tau1_k, eps_k, score_g, sched) -> tau_tilde_km1: ...
```

### 3.2 None-default injection (the zero-risk mechanism)

`transport` is set at **construction time** (static), so the guard is a
Python-level branch, not a traced `lax.cond`. When `None`, the original inline
lines are traced unchanged → identical graph → identical floats:

```python
if self.transport is None:
    <verbatim existing DDPM lines>          # byte-identical default
else:
    tau_tilde = self.transport.step(tau_k, tau1_k, eps_k, score_g, sched)
```

> **Rule:** never "refactor the default into a function that is mathematically
> equal". Keep the default lines verbatim; only *add* the `else`. This is what
> makes the regression test pass byte-for-byte.

### 3.3 Family coefficient table

All three are `τ̃_{k-1} = sqrt(abar_{k-1}) τ̂_{1|k} + c_k ε̂_k + σ_eff P ξ`,
differing only in `c_k` and how `σ` enters (`latex_2go/sections/prelims.tex`):

| family | deterministic map `c_k` (coeff on `ε̂_k`) | `σ_eff` | latex eq |
|---|---|---|---|
| DDPM | score-form (`mbd_jax:269-271`); equivalently `sqrt(1-abar_{k-1})` with `σ` re-injected | `≥0` (here 0) | `eq:ddpm_reverse_2go` |
| DDIM | `sqrt(1 - abar_{k-1} - σ_k²)` | `≥0` (det. at 0) | `eq:ddim_update_2go` |
| FM   | `sqrt(1 - abar_{k-1})` | `≥0` (det. at 0) | `eq:fm_update_expanded_2go` |

Note: **DDIM ≠ "σ=0"**. `σ_eff` is an independent stochasticity knob shared by
all families; the family is the deterministic recombination. The 2GO geometry
drift `Ŝ_{G,k}` and tangent noise `P ξ` compose on top, unchanged.

### 3.4 `ρ_k` schedule

`DiffusionScheduler.diffusion_params(state)` gains two fields:
`transport_family` (`"DDPM"|"DDIM"|"FM"`) and the per-family scalars it needs.
Per-step switching in the traced `scan` uses `jax.lax.switch(family_idx,
[ddpm_fn, ddim_fn, fm_fn], args)`; a continuous DDPM↔DDIM↔FM **blend** (if
wanted) is just a convex combination of the `c_k` coefficients and avoids the
switch entirely.

---

## 4. Zero-risk guarantee: None-default + regression

1. **Before any change**, snapshot a baseline per solver+config:
   run with the current code, save the final trajectory (and a few diffusion
   intermediates) to `tests/baselines/<solver>_<config>.npz`.
2. **After** adding the transport seam (still `transport=None` default), re-run
   and assert **exact** equality (`np.array_equal`, or `atol=0` /
   `rtol=0` allclose) against the baseline.
3. Only then wire DDIM/FM behind explicit config.

If step 2 ever fails, the default path was perturbed — fix before proceeding.
This is the same discipline as the `noise_sampler` rollout that already shipped.

---

## 5. Implementation phases

- **P0 — Baselines.** Add `tests/baselines/` + a generator script; capture
  current outputs for mbd / ebmbd / mdoc / mdcoas / cfsmbd / 2go on their smoke
  configs.
- **P1 — Interface + DDPM identity.** Add `ReverseTransport` (+ jax/numpy
  backends), a `DDPMTransport` whose `step` reproduces the verbatim default, and
  the None-default guard in **one** solver (mbd). Regression must pass with both
  `transport=None` and `transport=DDPMTransport()` (the latter proves the DDPM
  backend is exact).
- **P2 — DDIM / FM backends.** Add `DDIMTransport`, `FMTransport` (coefficient
  recombination from §3.3) + materialise `ε̂_k`. Unit-test each family's `c_k`
  math in isolation.
- **P3 — Scheduler emits `ρ_k`.** Extend `DiffusionScheduler` (fixed +
  dualcontrol) to emit `transport_family`; add `lax.switch` (or blend) in the
  scan. Config-driven.
- **P4 — Roll out to remaining solvers.** Repeat P1's None-default guard in
  ebmbd / mdoc / mdcoas / cfsmbd / 2go, each gated + regression-locked.
- **P5 (later) — Single-pass Schrödinger Bridge.** Add `SBTransport` (one extra
  diffusion coefficient `g_k` + reference drift); single reverse pass only.

---

## 6. Per-solver rollout

Each solver has its own `scan` body with solver-specific surroundings; the
transport seam is only the **innermost** `(τ_k, τ̂, ε̂, Ŝ) → τ̃` recombination.
Everything else (rollout, weighting, energy barrier, CFS filter, geometry)
stays put.

| solver | reverse-body file | extras around the transport |
|---|---|---|
| mbd | `mbd/backends/mbd_jax.py` | none (canonical) |
| ebmbd | `ebmbd/backends/ebmbd_jax.py` | energy-barrier prior |
| mdoc | `mdoc/backends/*` | — |
| mdcoas | `*` | — |
| cfsmbd | `cfsmbd/backends/cfsmbd_jax.py` | CFS QP retraction |
| 2go | `twogo/backends/twogo_jax.py` | geometry G/P + CFS + gate |

Order: mbd (P1) → cfsmbd → 2go → ebmbd/mdoc/mdcoas (P4).

---

## 7. Files

**New**
- `genedynamics/solvers/.../transport/base.py` — `ReverseTransport` + family enum.
- `genedynamics/solvers/.../transport/backends/{ddpm,ddim,fm}_jax.py`.
- `tests/baselines/` + `scripts/.../capture_transport_baselines.py`.
- `tests/test_transport_regression.py` — byte-identical default guard.
- `tests/test_transport_families.py` — per-family `c_k` math.

**Edited (additively, behind None-default)**
- each backend `*_jax.py`: materialise `ε̂_k`, add the `if transport is None`
  guard.
- `DiffusionScheduler/base.py`, `fixed/`, `dualcontrol/`: emit
  `transport_family`.

---

## 8. Testing

1. **Regression (the safety net):** default `transport=None` ⇒ exact match vs
   baseline, every solver.
2. **DDPM-backend identity:** `transport=DDPMTransport()` ⇒ exact match vs the
   `None` baseline (proves the explicit DDPM backend equals the inline default).
3. **Family math:** unit-test `c_k` for DDIM/FM against the latex formulas on
   synthetic `(τ_k, τ̂, ε̂)`.
4. **Integration:** one full smoke run per family per solver; assert it
   completes and produces finite trajectories.

---

## 9. Risks & mitigations

| risk | mitigation |
|---|---|
| Default path perturbed (results change) | verbatim default + byte-identical regression test (§4) |
| `ε̂_k` divide-by-zero near `abar→1` | clamp `sqrt(1-abar)` with `eps`; only used by DDIM/FM |
| `lax.switch` changes graph/FP even when family=DDPM | keep DDPM as the inline default; switch only added when a non-DDPM family is configured |
| Six duplicated reverse bodies drift | shared `ReverseTransport.step`; the per-solver edit is only the guard |
| SB IPF outer loop breaks single `scan` | scope SB to single-pass only (P5) |

---

## 10. Open decisions

1. **Per-step family selection vs continuous blend.** `lax.switch` (discrete
   `ρ_k`) or convex-blended `c_k` (smooth schedule). Blend avoids the switch and
   matches "adaptive schedule between families".
2. **Where the transport lives.** A shared `solvers/.../transport/` package
   (preferred) vs per-solver. Shared keeps the six bodies from drifting.
3. **`σ_eff` ownership.** For 2GO the gate already produces `σ_eff`; for the
   plain MBD family `σ_eff = extra_sigma`. Confirm the transport consumes a
   single `σ_eff` input rather than recomputing.
