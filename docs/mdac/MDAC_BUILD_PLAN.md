# MDAC Build Plan — From DIAL to Manifold Diffusion Annealing Control

> **Status (2026-06-22).** DIAL is fully ported + verified as a standalone multi-backend solver. The shared receding-horizon bridge (`RecedingHorizonController` + `WarmStartPlanner` + `SolverPlannerAdapter`), `env_rollout`, node-spline, and the reverse-strategy `update_form` seam are all built and tested. **This document is forward-looking: "how to build MDAC from here."** It does NOT re-derive DIAL. The source of truth for the MDAC algorithm is `idea.txt` (the paper).
>
> **Supersedes** `MDAC_PLAN.md` and `MDAC_GENEDYNAMICS_PLAN.md`. The earlier "MDAC = `CFSMBDSolver` subclass + `CFSMBDBackendJax` overlay" framing is **VOID** (see §3). The per-component math from `MDAC_PLAN.md` is preserved in **Appendix A**.

---

## 1. What MDAC is

**Thesis.** MDAC (Manifold Diffusion Annealing Control) is a receding-horizon sampling-based controller that samples a **lower-control sequence** `U = u_{0:H-1}` (never a state trajectory — states come from closed-loop rollout `s_{h+1}=F(s_h,u_h;c)`, `idea.txt` eq:closed_loop_rollout) from the target

```
p*(U | s_0, c) ∝ p_ψ(U|s_0,c)^{λ_ψ} · exp(−J_cl(U;s_0,c)/β) · 1_{M(s_0,c)}(U)      (eq:joint_target_hard)
```

i.e. a **model-free RL prior** `p_ψ`, a **model-based closed-loop Boltzmann likelihood** `exp(−J_cl/β)`, and a **hard indicator on the contact/safety manifold** `1_M`. MDAC is DIAL's MBD annealing made **manifold-aware**: it adds five upgrades over DIAL, each grounded in `idea.txt`:

1. **Position-stiffness primitive** (eq:position_stiffness_u, eq:stiffness_log_param): `u_h=(r_h,K_h,ν_h)`, with `K_h=exp(S_h)` (log-Euclidean SPD) so sampling/optimization happens over symmetric `S_h` and executed `K_h` is SPD by construction.
2. **Soft-feasibility / augmented-Lagrangian weighting** (eq:soft_feasibility_cost): the hard indicator is relaxed to `J̃_k = J_cl + λ_k^T r + (ρ_k/2)‖r‖²` with residual `r=[h ; [g]_+]` during scoring; hard feasibility is restored by retraction after the update.
3. **Constraint-manifold geometry** (eq:method_metric, eq:geometry_shaped_score, eq:tangent_noise, eq:method_retraction): a task-aware metric `G_k = I + ρ_k J_C^T W_C J_C + κ_k J_tr^T W_tr J_tr + G_K`, a metric-aware tangent projection `P_M^{G_k} = I − G^{-1}J_C^T(J_C G^{-1}J_C^T)^† J_C` applied to **both** the score and the noise, and a **linearized retraction QP** that projects the raw update back onto the manifold.
4. **Model-free RL prior `p_ψ`** (eq:rl_warm_start, eq:importance_weight): injected two ways — warm-start `U^init = λ_shift·U^shift + (1−λ_shift)·U^rl`, and a log-prior term `+λ_{ψ,k}·log p_ψ` in each candidate's log-weight. Optional full path: an elite buffer of optimized `U`s trains a conditioned diffusion prior `s_θ`, fused as `Ŝ_k = ω^mb Ŝ^mb + ω^mf s_θ` (minimal impl sets `ω^mf=0`).
5. **Generalized reverse update + coupled annealing** (eq:ddpm_control_update, eq:ddim_control_update, eq:coupled_schedule): one update covering DDPM (σ>0) / DDIM·flow (σ=0), driven by the coupled schedule `(σ_k↓, ρ_k↑, κ_k↑, λ_{ψ,k}↓)` — early steps explore via RL prior + noise, late steps enforce rollout cost + constraints + retraction.

**Research questions** (`idea.txt` lines 661-668):
- **RQ1** Does sampling lower-control primitives `U` beat trajectory-level / torque-only sampling in contact-rich tasks?
- **RQ2** Does the position-stiffness primitive improve the safety/task tradeoff vs fixed- or position-only stiffness?
- **RQ3** Does manifold tangent denoising + retraction reduce contact/force/balance violations vs sequential safety filters?
- **RQ4** Does the joint MF/MB formulation improve sample efficiency & generalization vs MF-only and MB-only?
- **RQ5** Can the same formulation handle both manipulation and humanoid loco-manipulation via different `(u_t, F, M)`?

---

## 2. Foundation already built (reuse inventory)

Everything below is built and verified. MDAC composes these — it does not rebuild them.

### 2.1 DIAL solver + node-spline (the diffusion engine)
- `genedynamics/solvers/single/dial/{dial.py, backend_impl.py, backends/dial_jax.py, spline.py, spline_cache.npz}`; registered name `"dial"`.
- `dial_jax.py::reverse_once` (L56) is the MBD weighted-mean update `Ybar = softmax(rews/temp)·Y0s` — **proven byte-identical to mbd DDPM score-form** and to dial-mpc upstream `dial_core.py:132`. It already carries an **`update_form` knob** (L68: `"weighted_mean" | "mbd_score"`) — the seed of a per-solver reverse-strategy registry that MDAC extends with an `"mdac"` form.
- `spline.py::NodeSpline` (`build/node2u/u2node`) — cached constant linear maps from `(Hnode+1, nu)` node controls to dense `(Hsample+1, nu)` controls. **MDAC does ALL geometry in node space** (small, dense, cheap; see §7).
- **MDAC reuses:** the entire sample→rollout→weight→update→anneal scan; node2u/u2node matmuls; `make_sigma_control`/`make_traj_diffuse_factors` (L44-53) sigma annealing.

### 2.2 Shared receding-horizon bridge + SolverPlannerAdapter
- `genedynamics/solvers/common/receding_horizon.py`:
  - `WarmStartPlanner` Protocol (L44) — `init_plan_var / replan / first_action / shift / make_schedule`.
  - `RecedingHorizonController` (L115) — the closed-loop MPC driver (execute first action, shift, replan).
  - `SolverPlannerAdapter` (L196) — wraps **any** solver's `solve()` into the bridge via re-solve; `warm_start_kwarg` enables warm-start when the solver's `solve` accepts an init-plan. **So "any solver + bridge" works without editing the solver.**
- `DialBackendJax` (`dial_jax.py` L258-281) **already implements the `WarmStartPlanner` protocol natively** (`init_plan_var/make_schedule/replan/first_action/shift`). This is the seam MDAC slots into: **MDAC's backend implements the same protocol natively** so it gets the bridge for free, and every baseline runs through the bridge via `SolverPlannerAdapter`.
- Test: `test/unit/test_solver_planner_adapter.py` (4 pass).
- **MDAC reuses:** warm-start scheduling, shift/replan/execute loop, fair baseline harness (all baselines = same bridge).

### 2.3 Shared brax rollout
- `genedynamics/solvers/common/env_rollout.py`: `is_brax_env` (L21), `build_brax_rollout` (L26), `build_brax_step` (L46) — shared brax-`State` rollout (reward extraction, transitions) reusable by all diffusion solvers.
- **MDAC reuses:** candidate rollout `s̃_{h+1}^m=F(s̃_h^m, ũ_{k,m,h};c)` (eq:method_candidate_rollout) — unchanged from DIAL.

### 2.4 Brax DIAL tasks (constrained-task substrate)
- 6 self-contained brax `PipelineEnv` tasks, no dial-mpc runtime: `humanoid_h1_{walk,loco,push_crate}`, `quadruped_go2_{walk,seq_jump,crate_climb}`; base `genedynamics/envs/legged_brax_base.py`; assets vendored under `genedynamics/envs/assets/dial/`; registered via `factories`.
- Verified byte-faithful: self-contained `make_env` + DIAL `h1_loco` `dx=+0.458` vs dial-mpc source `0.484` (residual = bridge execute/shift/diffuse order + RNG, ~2-5%, NOT algorithmic).

### 2.5 Geometry / transport / AL / retraction / schedule machinery
| MDAC need | Reuse | Path / symbol |
|---|---|---|
| Tangent projection `P_M^G` (Woodbury/Cholesky) | genemetry | `genemetry/manifold/sdf.py::SdfManifold.project`; `genemetry/ops/backends/jax_ops.py::project_complement_batch` (L51), `build_active_rows` (L24) |
| Retraction `Retr_M` (linearized QP) | genemetry | `genemetry/retraction/cfs.py::CfsRetraction.retract`; `local_cfs.py` |
| Coupled annealing `(σ↓,ρ↑,κ↑)` | genemetry | `genemetry/schedule/overlay.py::ScheduleOverlay.compute`; `schedule/config.py::OverlayConfig` |
| Geometry-shaped score injection | transport | `solvers/common/transport/base.py::ReverseTransport.step` reserved `score_g` seam (None everywhere today — **MDAC is its first consumer**); 2GO scan `twogo_jax.py` as reference analog |
| DDPM/DDIM/FM updates | transport | `transport/backends/{ddpm_jax.py,ddim_jax.py,fm_jax.py}`; per-step `sched` dict `{abar_k, abar_km1, alpha_k}` |
| Augmented-Lagrangian rollout | cfsmbd | `cfsmbd/backends/cfsmbd_jax.py::rollout_augmented_reward_and_v` (L364), `aug_lambda/aug_rho`, `ALMAdaptiveScheduler` |
| RL prior net (jit-able) | SHAC | SHAC JAX `init_mlp`/`mlp_apply`; deploy `PolicyArtifact`; `mrmfmbd theta_prior` add-to-logw pattern |
| Experiment/baseline/metrics | plugins | `MethodPlugin`/`MetricsPlugin`/`EnvironmentPlugin`/`BaselineProtocol`; `corridor._cvar`; `ExperimentRunner.run_all`; pre-registered MPPI/CEM/DIAL/2GO |

### 2.6 ⚠️ VALIDATED FINDING that drives the whole design

**On the UNCONSTRAINED dial locomotion tasks, every MBD-family solver degenerates to the SAME MBD weighted mean == DIAL** (mbd/cfsmbd/mdoc/2go byte-identical, e.g. `go2_walk dx=+0.54074` for all; ebmbd same). Their distinguishing ops are **inert without a constraint manifold**: cfsmbd CFS-QP → no-op; mdoc filter → no-op; 2go SDF geometry → identity (rank-0); ebmbd log-barrier → 0.

**Consequence:** MDAC's value (and every constrained solver's) shows ONLY on **constrained, contact-rich tasks**. The manifold must have nonzero `J_C` for `P_M^G` to differ from identity. **All MDAC experiments and golden-difference gates must be on constrained tasks** (corridor / stepping-stones / the two `idea.txt` experiments / constraint-augmented dial tasks with an SDF manifold). On unconstrained tasks MDAC must reduce **byte-identically to DIAL** (this is a regression gate, not a result).

---

## 3. MDAC architecture (corrected)

**MDAC is NOT a `CFSMBDSolver` subclass and NOT a `CFSMBDBackendJax` overlay.** That framing (in the old `MDAC_GENEDYNAMICS_PLAN.md`) is void. MDAC is a **new, self-contained multi-backend solver** under `genedynamics/solvers/single/mdac/`, mirroring the DIAL layout, deriving its reverse kernel from `dial_jax.py` (MBD weighted-mean engine) and adding the five `idea.txt` components as composable layers.

### 3.1 How it slots in
- MDAC's backend `MdacBackendJax` **implements the `WarmStartPlanner` protocol natively** (exactly as `DialBackendJax` does, §2.2) → gets `RecedingHorizonController` for free; every baseline still runs via `SolverPlannerAdapter`.
- MDAC's reverse step registers a new reverse strategy `"mdac"` alongside DIAL's `update_form ∈ {weighted_mean, mbd_score}`. The strategy registry is the single dispatch point; the **shared rollout is untouched** — only post-rollout score/update/safety differ.
- The reverse step is the canonical `idea.txt` Algorithm-1 inner loop: **MB MC score (DIAL engine) → joint score with RL log-prior → `G_k`-preconditioned tangent-projected score + tangent noise → DDPM/DDIM/flow raw update → linearized retraction QP → re-pin node-0**, all in **node space**.

### 3.2 The MDAC reverse step (canonical reference, node space)
```python
# strategy == "mdac"  (one reverse step k; everything below is node-space, shapes ~ (Hnode+1, nu))
Y0s, eps = sample_candidates(rng, Ybar_i, abar_k, noise_scale)   # mean = Ybar/√ᾱ  (eq:method_clean_proposal)
us       = node2u_vvmap(Y0s)                                     # NodeSpline.node2u, constant matmul
rewss, pss = rollout_us_vmap(state, us)                          # SHARED env_rollout, unchanged
h, g     = vmap(spec.residual_fn)(pss, us, c)                    # clean-state residuals (NO physics backprop)
r        = concat([h, relu(g)])                                  # eq:constraint_residual
Jtilde   = (-rewss.mean(-1)) + sched.lam @ r.T + 0.5*sched.rho*sum(r**2,-1)   # eq:soft_feasibility_cost
logw     = -(1/beta)*Jtilde + sched.lam_psi*logp_psi            # eq:importance_weight  (logp_psi minus batch-max)
w        = softmax(logw);  Ubar = einsum('n,nij->ij', w, Y0s)   # eq:method_clean_mean
S_mb     = (-Uk/(1-ᾱ) + √ᾱ/(1-ᾱ)*Ubar).ravel()                 # eq:method_mb_score   (clip 1-ᾱ from 0)
S_hat    = omega_mb*S_mb + omega_mf*s_theta(...)                # eq:joint_score (omega_mf=0 minimal)
JC       = jacrev(spec.reduced_eq)(Ubar.ravel(), state, c)      # clean-state J_C, zero physics backprop
G        = build_metric(JC, Jtr, sched.rho, sched.kappa, W_C, W_tr, G_K_diag)   # eq:method_metric
S_M      = geometry_shaped_score(S_hat, JC, G, proj_reg)        # P·G⁻¹·Ŝ  (eq:geometry_shaped_score)
epsM     = tangent_noise(eps, JC, G, proj_reg)                  # eq:tangent_noise
U_raw    = ddpm_or_ddim(Uk, S_M, epsM, ᾱ_k, ᾱ_{k-1}, sched.sigma, ddim_eta)   # eq:ddpm/ddim_control_update
U_km1    = linearized_retraction(U_raw, ...)                    # eq:linearized_retraction (≤2 steps)
Ybar     = U_km1.reshape(...).at[0].set(Ybar_i[0])             # re-pin node-0
```

### 3.3 File layout (new)
```
genedynamics/solvers/single/mdac/
  __init__.py
  mdac.py                      # thin solver shell, register_solver("mdac"), config surface (DIAL-default fields)
  backend_impl.py              # to_unified_backend; DialBackend-style Protocol conformance
  backends/
    mdac_jax.py                # MdacBackendJax: WarmStartPlanner native + reverse strategy "mdac"
  core/
    schedule.py                # coupled (σ_k,ρ_k,κ_k,λ_{ψ,k},ᾱ_k) ScheduleStep pytree; wraps genemetry ScheduleOverlay
    feasibility.py             # stack_residual, soft_feasibility_cost, importance_logw, mb_score, al_update, adaptive_beta
    manifold_geometry.py       # build_metric, geometry_shaped_score, tangent_noise, linearized_retraction (wrap genemetry)
    constraint_spec.py         # ConstraintSpec + reduced_constraints; jac_mode ∈ {clean_state, reduced_autodiff, fd_node}
    primitive_layout.py        # PrimitiveSpec, svec2sym/sym2svec, stiffness_log_to_pd (eigh expm), unpack_primitive, metric_GK
    strategy_base.py           # ReverseStrategy registry (mppi | dial | mdac | pegasus)
    method_registry.py         # MethodFlags + METHOD_TABLE + resolve_method (fairness self-check: same (M,H,K))
  priors/
    rl_prior.py                # RLPrior: act/rollout/logp_of_sequence from saved meta (SHAC MLP, jit-safe)
    diffusion_prior.py         # s_θ conditioned diffusion (omega_mf>0); via transport score_g seam
    elite_buffer.py            # elite optimized-U buffer → s_θ training data
  envs/                        # MDAC constrained envs live here or under genedynamics/envs/ (see §6)
```

---

## 4. The 5 components to build

For each: **math (eq)** · **reuse** · **new (file)** · **gate**.

### 4.1 Position-stiffness SPD primitive
**Math.** `u_h=(r_h,K_h,ν_h)` (eq:position_stiffness_u); `K_h=exp(S_h)`, `S_h=S_h^T` (eq:stiffness_log_param) — sample over symmetric `S_h`, execute SPD `K_h`. Low-level map `a_h=π_low(s_h,u_h;c)` (impedance/WBC) then `F=f(s_h,a_h)` (eq lines 26-34). `action_size` (= `PrimitiveSpec.total_width`) is the single coupling point that widens the diffusion tensor.
**Reuse.** `env.action_size → MBDPI.nu` auto-coupling; SHAC eigh patterns; genemetry block-diagonal metric assembly.
**New.** `mdac/core/primitive_layout.py`: `svec2sym/sym2svec` (round-trip), `stiffness_log_to_pd` (`K=expm(S)` via `eigh`), `unpack_primitive`, `metric_GK_nodewise` (`G_K=blkdiag(0,…,W_S,…,0)`, eq line 471-475); `act2impedance_tau` in the env's impedance branch.
**Gate (Track A).** `svec2sym(sym2svec(M))==M` for d∈{3,6}; `K=expm(...)` SPD and `jax.grad` finite; `action_size` regression — torque/position envs unchanged, impedance env returns `total_width` and `MBDPI(env).nu` catches the widened width.

### 4.2 Soft-feasibility (augmented-Lagrangian) weighting
**Math.** `J̃_k=J_cl+λ_k^T r+(ρ_k/2)‖r‖²` (eq:soft_feasibility_cost); `r=[h ; [g]_+]` (eq:constraint_residual); `log w_{k,m}=−(1/β_k)J̃_k+λ_{ψ,k}log p_ψ` (eq:importance_weight); softmax weights; `β=temp·rews.std()`. **Sign trap:** `J_cl=−rews.mean(-1)` (DIAL maximizes reward, AL is a cost). `λ_k`,`ρ_k` annealed across reverse steps (early lenient, late strict).
**Reuse.** `cfsmbd_jax.py::rollout_augmented_reward_and_v` (L364), `aug_lambda/aug_rho`, `ALMAdaptiveScheduler`.
**New.** `mdac/core/feasibility.py`: `stack_residual`, `soft_feasibility_cost`, `importance_logw`, `mb_score`, `al_update`, `adaptive_beta`. `base_env.constraint_residual` default-empty + per-env override.
**Gate (Track A unit; Track B e2e).** `use_soft_feasibility=False` allclose with DIAL; `h=g=0 ⇒ J̃==J_cl`; relu gating; violation ⇒ `J̃↑` & weight↓; `al_update` ρ geometric growth. E2e: residual `‖r‖` decreases within reverse steps; foot penetration below DIAL; `Nsample=2048` no NaN.

### 4.3 Constraint-manifold geometry: metric `G_k` + tangent projection `P_M^G` + retraction
**Math.** `G_k=I+ρ_k J_C^T W_C J_C+κ_k J_tr^T W_tr J_tr+G_K` (eq:method_metric); `Ŝ_{M,k}=P_M^{G_k}(U_k)·G_k^{-1}·Ŝ_k` (eq:geometry_shaped_score); `P_M^G=I−G^{-1}J_C^T(J_C G^{-1}J_C^T)^† J_C` (eq:metric_tangent_projection); `ε_{M,k}=P_M^{G_k}ε_k` (eq:tangent_noise); retraction QP `argmin ½‖U−U^raw‖²_{G_k} s.t. C=0, K=exp(S)` solved by 1-2 linearized steps (eq:linearized_retraction). **`J_C` is evaluated on clean predicted states** (`jac_mode='clean_state'`): differentiate only through the constant spline matrix + analytic kinematics/contact Jacobian — **never through the mjx rollout**.
**Reuse.** `genemetry/ops/backends/jax_ops.py::project_complement_batch` (Woodbury/Cholesky), `build_active_rows`, top-k active rows; `genemetry/retraction/cfs.py::CfsRetraction.retract`; `genemetry/schedule/overlay.py::ScheduleOverlay`; 2GO `twogo_jax.py` geometry-shaped reverse as the working analog; transport `score_g` seam.
**New.** `mdac/core/manifold_geometry.py`: `build_metric`, `geometry_shaped_score` (P·G⁻¹·Ŝ via Woodbury/Schur + Cholesky + Tikhonov `proj_reg`, solve not pinv), `tangent_noise`, `linearized_retraction`, `spd_woodbury_solve`. `mdac/core/constraint_spec.py`: `ConstraintSpec`+`reduced_constraints`, `jac_mode` (`clean_state` default, `reduced_autodiff`/`fd_node` fallbacks); hybrid-mode static `mode_mask` + host-side `select_mode`.
**Gate (Track A).** projection idempotent `‖P(P(v))−P(v)‖<1e-5`; tangency `‖J_C P(v)‖<1e-5`; Woodbury==dense-pinv (1e-5); `G_K` only moves S coords; retraction lowers `‖C‖`; `use_manifold=False` byte-identical DIAL; no-op spec == weighted mean; `DDPM(σ=0)==DDIM(η=0)`. **(Track B)** **zero `pipeline_step` in the jaxpr** of the `J_C` computation.

### 4.4 Model-free RL prior `p_ψ`
**Math.** Warm-start `U^init=λ_shift·U^shift+(1−λ_shift)·U^rl` (eq:rl_warm_start); log-prior term `+λ_{ψ,k}·log p_ψ` (eq:importance_weight); prior `p_ψ(U)=Π_h π_ψ(u_h|ô_h,c)` (eq:rl_prior_horizon). Optional `s_θ`: elite optimized-`U` buffer → conditioned diffusion → `Ŝ_k=ω^mb Ŝ^mb+ω^mf s_θ` (eq:joint_score; minimal `ω^mf=0`). PPO (humanoid) / SAC (manipulator).
**Reuse.** SHAC JAX MLP (`init_mlp`/`mlp_apply`); deploy `PolicyArtifact`; `mrmfmbd theta_prior` add-to-logw; transport `score_g` seam for `s_θ`.
**New.** `mdac/priors/rl_prior.py` (`RLPrior`: act/rollout/logp_of_sequence from saved meta, jit-safe Gaussian+tanh logp); `_get_obs_from_pipeline` info-free obs path (so `logp_of_sequence` is jit-safe); `mdac/priors/diffusion_prior.py`; `mdac/priors/elite_buffer.py`; `train_rl_prior.py`. Warm-start mix + `λ_ψ` schedule wired into `replan`/reverse scan.
**Gate (Track A logp math; Track B e2e).** `use_rl_prior=False, λ_ψ0=0, ω_mf=0` bitwise-equal to §4.2 result; `jit(reverse_once)` compiles with `use_rl_prior=True` (proves obs has no `info` dependency); warm-start endpoints (`λ_shift=0→U_rl`, `=1→U_shift`); `logp` minus batch-max no collapse. **`s_θ` strictly after minimal-prior elite collection.**

### 4.5 Generalized reverse (DDPM/DDIM/FM) + coupled annealing
**Math.** Clean proposal `Ũ~N(U_k/√ᾱ_k,(1/ᾱ_k−1)Σ_k)` (eq:method_clean_proposal — the `√ᾱ` rescale DIAL lacks). DDPM raw update `U_{k-1}^raw=(1/√α_k)(U_k+(1−ᾱ_k)Ŝ_{M,k})+σ_k ε_{M,k}` (eq:ddpm_control_update); DDIM/flow via predicted clean `Û_{1,k}` (eq:ddim_control_update); `σ_k>0` diffusion, `σ_k=0` flow. Coupled schedule `(σ_k↓, ρ_k↑, κ_k↑, λ_{ψ,k}↓)` (eq:coupled_schedule, eq:schedule_trend), `ᾱ_k∈(0,1]↑`, `1-ᾱ` clipped from 0.
**Reuse.** transport `ddpm_jax/ddim_jax/fm_jax` + `sched` dict; DIAL `make_sigma_control`/`make_traj_diffuse_factors`; genemetry `ScheduleOverlay`/`OverlayConfig`.
**New.** `mdac/core/schedule.py`: `ScheduleStep` (7-leaf pytree: `σ,ρ,κ,λ_ψ,ᾱ_k,ᾱ_{k-1},lam`), `build_schedule` (synthesizes DIAL's missing `ᾱ` base, couples genemetry overlay). Unified `ddpm_or_ddim(U_k,S_M,epsM,ᾱ_k,ᾱ_{k-1},σ,ddim_eta)` in `strategy_base`.
**Gate (Track A).** `DDPM(σ=0)==DDIM(η=0)`; schedule monotone trends asserted; `1-ᾱ` clipped; with all-default (DIAL) config, reverse reduces to DIAL weighted mean byte-identically.

---

## 5. Milestone plan

Current state: **DIAL done, bridge done, env_rollout done, node-spline done, reverse-strategy seam done.** Track A = hardware-free (fedguide x86, unit-testable, no mjx). Track B = needs native arm64 / GPU (mjx rollout). **MPPI/DIAL byte-golden is a BLOCKING regression at every milestone.**

| M | Goal | Track | Exit criterion |
|---|---|---|---|
| **M0** | Scaffold `solvers/single/mdac/`; `register_solver("mdac")`; DIAL-default config fields; `ReverseStrategy` registry (mppi/dial reproduced); `ScheduleStep`; freeze golden `.npy` (`go2_walk`, `h1_loco`). | A (scaffold) + B (golden freeze) | golden `us[]/rews[]` within 1e-5 of DIAL with all MDAC switches off; registry mppi/dial parity. |
| **M1** | §4.1 primitive (`primitive_layout.py`) + impedance env branch as wiring proof. | A | svec round-trip; `K=expm` SPD + grad finite; `action_size` regression. |
| **M2** | §4.5 schedule (`schedule.py`) + generalized reverse in `strategy_base`. | A | `DDPM(σ=0)==DDIM(η=0)`; default config == DIAL byte-identical. |
| **M3** | §4.2 feasibility (`feasibility.py`) + `base_env.constraint_residual`. | A (unit) | sign/relu/violation tests; `use_soft_feasibility=False` allclose DIAL. |
| **M4** | §4.3 geometry (`manifold_geometry.py`, `constraint_spec.py`, hybrid mode). **Core math.** | A (math) + B (jaxpr) | idempotence/tangency/Woodbury==pinv/`G_K`/retraction-lowers-`‖C‖`; `use_manifold=False` byte-identical DIAL; **zero `pipeline_step` in `J_C` jaxpr**. |
| **M5** | Wire M2-M4 into `mdac_jax.reverse_once` (the §3.2 step); `MdacBackendJax` implements `WarmStartPlanner`; run through `RecedingHorizonController`. | B | On a **constrained** task, MDAC differs from DIAL (manifold active); on **unconstrained** dial task, byte-identical DIAL; no NaN at `Nsample=2048`. |
| **M6** | §4.4 RL prior `RLPrior` + warm-start mix + `λ_ψ` schedule; then PPO/SAC train + elite buffer + `s_θ`. | B (env) + A (logp math) | `use_rl_prior=False` bitwise-equal M5; `jit` compiles with prior on; warm-start endpoints; `s_θ` after elite collection. |
| **M7** | Exp envs (§6): humanoid box-push SE(2) + arm surface-scan; `surface_geometry.py` (NURBS). | B (envs) + A (NURBS math) | face one-hot→contact point; residual sign tests; `J` monotone toward goal; `mppi` 50 steps moves toward goal, fall_rate=0, no NaN. |
| **M8** | Baseline/ablation harness (`method_registry.py`) + eval (metrics, CVaR95, violation rates, RQ1-5 sweep, bootstrap CI, LaTeX tables). | A (registry/metrics) + B (runs) | fairness self-check (same `(M,H,K)`); 8 ablations each bypass exactly one component; metrics unit tests; aggregate emits table + plots. |

**Critical path:** `M0 → {M1 ∥ M2 ∥ M3 ∥ M4} → M5 → M6 → M7 → M8`. M1-M4 are Track-A-heavy and parallelizable (≈45% of load-bearing code is hardware-free). M6's `s_θ` strictly follows minimal-prior elite collection. M8 last (needs registered MDAC + envs + RL baselines).

---

## 6. Experiments & baselines

**Where MDAC differs from DIAL/MBD/2GO/cfsmbd: only on constrained, contact-rich tasks** (§2.6). Two task families from `idea.txt`, plus the existing constrained corridor/stepping-stones as fast iteration substrates.

- **Exp I — Surface-contact manipulation** (`idea.txt` 670-782): 7-DoF arm scanning NURBS surfaces. Primitive `u^arm=(Δξ,Δη,Δψ,S_h,F^d_n)`; impedance law; constraints surface-attach `h_surf`, normal-align `h_normal`, force bounds `g_force`; surfaces S1-S4 (planar→unseen NURBS). Metrics: surface/normal/force tracking, force violation rate, **force CVaR95**, contact loss, coverage, stiffness smoothness, energy, runtime.
- **Exp II — Humanoid box pushing** (`idea.txt` 784-954): humanoid pushes SE(2) box to goal under hand-object + foot-ground contact, balance, friction, non-tip. Hybrid manifold `M_hum` (box-ground/hand-box face-select/stance-foot); inequalities `g_bal,g_fric,g_tip`. Levels H1-H4 (double-support → walk-and-push → unjamming with face selection). Metrics: box-goal success, fall rate, hand contact loss/slip, friction violation, tip/balance margin, force CVaR95, energy.

**Baselines** (`idea.txt` 961-980), **all run through the shared bridge via `SolverPlannerAdapter`** (no per-baseline plumbing): PPO/SAC (±learned stiffness), SRL-VIC, ATACOM-style RL, MPPI, DIAL-MPC-style annealing, PegasusFlow/WBFO sampling, RL+CBF/ISSA. MDAC is the only one with all of {MF prior, MB rollout, manifold safety}. **Fairness:** all sampling methods optimize the same `U`, same `F`, same `(M,H,K)`; DIAL/MPPI ablate tangent-projection+retraction but keep the sample budget (enforced by `method_registry` self-check).

**Ablations** (`idea.txt` 993-1011), each bypassing exactly one component: w/o stiffness · fixed stiffness · w/o RL prior · w/o MB rollout · w/o tangent projection · w/o retraction · w/o adaptive schedule · Euclidean stiffness (vs log-SPD). Each maps to one config flag in `MethodFlags`.

**Metrics framing:** report **task success** AND **constraint/contact/force/balance violation** (rate + CVaR95) — the latter is where MDAC beats sequential safety filters (RQ3) and fixed stiffness (RQ2).

---

## 7. Correctness traps

1. **Reward/cost sign trap.** DIAL *maximizes* reward; AL is a *cost*. If the sign flips, the optimizer *rewards* constraint violation (silent). Fix: single documented `J_cl=−rews.mean(-1)`, applied once; unit test "violation ⇒ `J̃↑` & softmax weight↓"; `β=temp·rews.std()` reproduces MPPI.
2. **Relaxed one-hot contact-face selection.** Discrete `j,a,b` is non-differentiable and breaks jit shapes. Fix: `softmax(face_logits/τ_face)` annealed to hard one-hot; H1-H3 fixed face (small τ), H4 anneals; never `argmax` inside jit; project the mixed contact point to the nearest face afterward.
3. **mjx exposes no contact force.** mjx gives only `contact.dist/pos`. Fix: quasi-static proxy — normal force ≈ commanded `F_n^d`; tangential ≈ `m_box·‖a_box‖` + ground friction; or project `cfrc_ext`; return NaN (don't crash) until wired; validate against non-jit MuJoCo offline before trusting RQ2/RQ3. *(Open: whether the proxy suffices for force-CVaR95/friction/tipping RQs or real `cfrc_ext` is mandatory.)*
4. **Never differentiate `J_C` through the mjx rollout.** 10-100× cost, non-smooth contact LCP, 2048 candidates infeasible. Architecturally forbidden; default `jac_mode='clean_state'`; M4 asserts **no `pipeline_step` in the jaxpr**. All geometry in **node space** (`n_U≈60-120`, `n_C≈10-60`), projection/retraction applied **once to the aggregated single score**, never per-candidate, never pinv on the `Nsample` axis.
5. **Bridge execute/shift/diffuse order + RNG (the verified ~2-5% residual).** The DIAL self-contained run matched upstream up to a 2-5% non-algorithmic residual attributed to the bridge's execute→shift→diffuse ordering and RNG threading. When comparing MDAC to any external/upstream number, expect (and account for) this; the byte-golden gate is against the **in-repo DIAL**, not upstream, to keep it exact.
6. **Degeneration caveat (§2.6).** On unconstrained tasks MDAC *must* equal DIAL — that is correctness, not a null result. Any "MDAC vs DIAL" comparison reported as a result must be on a task with a non-trivial active manifold (nonzero `J_C`), or it is meaningless.

---

## Appendix A — Deep per-component math (reference)

> Preserved from `MDAC_PLAN.md`. Use as the equation-level reference when implementing §4.

### A.1 Position-stiffness primitive
`u=(r,K,ν)`; SPD via log-Euclidean `K=exp(S)`, `S` symmetric (matrix exp of symmetric coord), keeps `jax.grad` finite. `svec2sym/sym2svec` round-trip for d∈{3,6}; `K=expm(...)` via `eigh`. Low-level `π_low`: `act2impedance_tau` (joint-space fallback + overridable `_impedance_task_law`). Auto-coupling: `env.action_size` is the single width source → `MBDPI.nu`.

### A.2 Soft-feasibility / Augmented Lagrangian
`J̃_k=J_cl+λ_k^T r+(ρ_k/2)‖r‖²`; **`J_cl=−rews.mean(-1)`** (the documented sign convention); `r=concat([h, relu(g)])`; `log w=−(1/β)J̃+λ_ψ·log p_ψ`; `w=softmax(log w)`; `Ūbar=einsum('n,nij->ij',w,Y0s)`; `β=temp·rews.std()`; `λ` via reverse-scan carry (`al_update`), `ρ` geometric growth. Invariants: violation ⇒ `J̃↑` ⇒ weight↓; `h=g=0 ⇒ J̃==J_cl`.

### A.3 Constraint-manifold geometry
`JC=jacrev(spec.reduced_eq)(Ūbar.ravel(),state,c)` on **clean predicted states** (`jac_mode='clean_state'`), derivative only via constant spline matrix + analytic kinematics/contact Jacobian (zero physics backprop). `build_metric` combines `J_C, J_tr, ρ_k, κ_k, W_C, W_tr, G_K_diag`; `G_K` acts only on S coords. `geometry_shaped_score = P·G⁻¹·Ŝ` via Woodbury/Schur + Cholesky + Tikhonov `proj_reg` (solve, not pinv); score AND noise projected. `tangent_noise(eps,JC,G,proj_reg)`. `linearized_retraction` 1-2 step QP lowering `‖C‖` (`retraction_iters≤2`, statically unrolled). MB score `S_hat=−U_k/(1-ᾱ)+(√ᾱ/(1-ᾱ))·Ūbar` (`1-ᾱ` clipped). Hybrid `M=∪_m M_m` via static `mode_mask` + host `select_mode`. Exit props: idempotent `<1e-5`; tangency `<1e-5`; Woodbury==dense-pinv `1e-5`; retraction lowers `‖C‖`.

### A.4 Model-free RL prior (+ optional `s_θ`)
Warm-start `U_init=λ_shift·U_shift+(1−λ_shift)·U_rl`; likelihood `+λ_ψ·log p_ψ` (`logp_psi` via vmap, minus batch-max); `RLPrior` thin adapter (act/rollout/logp_of_sequence from saved meta; PPO humanoid / SAC arm); `_get_obs_from_pipeline` info-free (jit-safe). Optional `s_θ` (`ω_mf>0`): elite sequences → DDPM-trained conditioned diffusion → `Ŝ=ω_mb·Ŝ^mb+ω_mf·s_θ`.

### A.5 Generalized reverse + coupled annealing
Clean proposal mean `Ybar/√ᾱ` (the rescale DIAL lacks). Unified `U_raw=ddpm_or_ddim(U_k,S_M,epsM,ᾱ_k,ᾱ_{k-1},σ,ddim_eta)`; `DDPM(σ=0)==DDIM(η=0)`. `ScheduleStep` 7-leaf pytree `(σ↓,ρ↑,κ↑,λ_ψ↓,ᾱ∈(0,1]↑)`, `1-ᾱ` clipped. Node-0 re-pinned after update: `Ybar=U_{k-1}.reshape(...).at[0].set(Ybar_i[0])`.

### A.6 Open math questions
- (#3) Which reverse update is the paper's main config — DDPM(η=1) vs DDIM/flow(η=0)? The other becomes an ablation.
- (#4) Can the quasi-static force proxy support force-CVaR95 / friction-cone / tipping RQs, or is real `cfrc_ext` mandatory?

---

**Source of truth:** `idea.txt` (1024 lines). **This plan supersedes** `MDAC_PLAN.md` (→ Appendix A) and `MDAC_GENEDYNAMICS_PLAN.md` (cfsmbd-overlay framing VOID).
