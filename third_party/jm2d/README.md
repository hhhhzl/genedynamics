# JM2D — Joint Model-based Model-free Diffusion (CoRL 2025)

Reference checkout for *Joint Model-based Model-free Diffusion for Planning
with Constraints* (Jung et al., 2025). **Optional**: we don't depend on
their code at runtime — we re-implement the two practical takeaways inside
our MRMFMBD solver. Cloning is only useful if you want to sanity-check the
math or rerun their toy domain.

## Setup (only if you want their reference code)

```sh
cd third_party/jm2d
git clone https://github.com/jm2d-corl25/jm2d.git .   # placeholder URL — see paper site
```

Project page: https://jm2d-corl25.github.io/

## What we actually take from JM2D

1. **Multi-step inner denoising for clean-sample estimation** (their Alg 2 / Table 3).
   Implemented inside `genedynamics/solvers/single/mrmfmbd/backends/mrmfmbd_jax.py`
   as `inner_denoise_steps: u` (Phase 1.3).

2. **Joint diffuse over (x, k)** with interaction potential
   `V(x, k) = exp(-J(k|x)/λ) · 1[g(k|x) ≤ 0]` (their Eq 7). For us,
   `k` will be the alm_adaptive dual variables (λ, ρ, p, ε). Phase 2+; not
   touched in Phase 0/1.

Neither takeaway requires importing their codebase, so this stub stays empty
unless someone wants to read their reference implementation.
