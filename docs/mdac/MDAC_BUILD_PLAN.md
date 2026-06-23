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
| Prior injection seams (reused by the NEW shared priors pkg, §4.4) | core/prob, transport, mrmfmbd, SHAC | `core/prob/noise_sampler.py::NoiseSampler` (L42) consumed by mbd/mppi/cfsmbd/ebmbd backends via `noise_sampler=` + `_draw_unit_noise`; `transport/base.py::ReverseTransport.step` `score_g` seam (L62); `mrmfmbd/theta_prior.py::ThetaPrior` (L92, `log_prob_batch`/`sample`, add-to-logw at `mrmfmbd_posterior_jax.py:349`) — to be **lifted** to the shared pkg; SHAC `critic.py::init_mlp`/`mlp_apply` (L26-50) jit MLP |
| Experiment / baseline / metrics framework | experiments | `experiments/framework/base.py` (Method/Environment/Metrics/ObstacleGenerator/Visualization Plugins L19-273, `BaselineProtocol`); `framework/registry.py::PluginRegistry` (L12); `framework/experiment.py::ExperimentRunner.run_all/run_single_experiment` (L78-711); `framework/config.py::ExperimentConfig` (L22) YAML; `experiments/runner.py::register_all_plugins` (L191-266); pre-registered methods mbd/mdoc/ebmbd/cfsmbd/**2go**/mrmfmbd (MPPI/CEM/DIAL **not yet** MethodPlugins — see §6) |

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

### 3.3 File layout (IMPLEMENTED — reuse, do NOT hand-roll)

> **Hard lesson (2026-06-22).** The first attempt hand-wrote `feasibility.py /
> schedule.py / manifold_geometry.py / constraint_spec.py / strategy_base.py`
> under `mdac/core/`. **All deleted** — they duplicated upstream machinery. MDAC
> is a THIN compose-solver in the mdoc/twogo high-performance pattern; the math
> lives upstream and is reused. New construction goes upstream first, then the
> solver imports it.

```
genedynamics/solvers/single/mdac/
  __init__.py
  mdac.py            # MDACSolver: DIAL/mdoc-style shell, brax substrate + node-spline,
                     #   instantiates seams (transport/constraint_filter/noise_sampler),
                     #   register_solver("mdac"), solve() == RecedingHorizonController (bridge)
  backend_impl.py    # to_unified_backend; MdacBackend Protocol
  backends/
    mdac_jax.py      # MdacBackendJax: jit + lax.scan reverse-diffuse, vmap rollout over
                     #   Nsample (mdoc/twogo pattern). Reuses dial make_sigma_control /
                     #   make_traj_diffuse_factors / NodeSpline / env_rollout. Composes the
                     #   seams; seams off => byte-identical DIAL. Native WarmStartPlanner.
  core/
    method_registry.py   # MethodFlags + METHOD_TABLE single-flag ablations + assert_fair.
                         #   ONLY solver-local module (config, not reusable math).
```

**Everything else is REUSED, not rebuilt** (the seams `mdac_jax.py` composes):

| MDAC need | Upstream module reused | Pattern from |
|---|---|---|
| soft-feasibility / iALM | augmented reward `J − (λ·mean[g]₊ + ρ/2·mean[g]₊²)` via `aug_lambda/aug_rho` + the env `constraint_residual` hook (`env_rollout.build_brax_rollout_augmented`); optional `_batched_alm_adaptive` λ/ρ | **cfsmbd / mdcoas** (`rollout_augmented_reward_and_v`) — **NOT mdoc** |
| action-space ConstraintFilter (separate from iALM) | `core/constraints/action_filters` `ConstraintFilter`/`NoOpConstraintFilter` (CFS-QP/CBF projection) | **mdoc** (`apply_actions_batch` per step) |
| DDPM/DDIM/FM adaptive reverse | `solvers/common/transport/backends` `AdaptiveTransport` (+ ddpm/ddim/fm) | **mdoc/2go** `transport=` seam (None=verbatim) |
| metric + tangent projection + retraction | `genemetry` `SdfManifold.geometry/project`, `ops/jax_ops.{build_active_rows,project_complement_batch}`, `retraction/cfs.CfsRetraction` | **2go** (dense space, per-step vmap) |
| coupled annealing | `genemetry/schedule/overlay.ScheduleOverlay.compute → OverlayParams(kappa,delta,sigma,theta,eta)` | **2go** |
| candidate noise | `core/prob.NoiseSampler` | mbd/mdoc/cfsmbd `noise_sampler=` |
| **NEW** position-stiffness SPD primitive | `genedynamics/core/control/stiffness.py` (UPSTREAM, not solver-private) | — (the one genuinely-new piece) |

**Priors are NOT under `mdac/`.** The RL prior `p_ψ`, learned diffusion prior `s_θ`, and elite buffer are shared infra in `genedynamics/learning/priors/` (§4.4), consumed via a `prior=` seam (M6).

---

## 4. The 5 components — REUSE map (do NOT hand-roll)

> **The components are not "built" — they are composed from the upstream modules
> in the §3.3 table.** Below, "reuse" names the exact upstream call; "new" is
> empty for everything except §4.1 (the SPD primitive, which lives upstream in
> `core/control/`). The first attempt hand-wrote §4.2/§4.3/§4.5 under `mdac/core/`
> and was deleted — that math already exists in `core/constraints`, `genemetry`,
> and `transport`. Keep the math (eq) below as the equation-level reference for
> *what the upstream call must compute*, not as a spec for new files.

### 4.1 Position-stiffness SPD primitive
**Math.** `u_h=(r_h,K_h,ν_h)` (eq:position_stiffness_u); `K_h=exp(S_h)`, `S_h=S_h^T` (eq:stiffness_log_param) — sample over symmetric `S_h`, execute SPD `K_h`. Low-level map `a_h=π_low(s_h,u_h;c)` (impedance/WBC) then `F=f(s_h,a_h)` (eq lines 26-34). `action_size` (= `PrimitiveSpec.total_width`) is the single coupling point that widens the diffusion tensor.
**Reuse.** `env.action_size → MBDPI.nu` auto-coupling.
**New — UPSTREAM (done):** `genedynamics/core/control/stiffness.py` (NOT solver-private): `svec2sym/sym2svec` (round-trip), `stiffness_log_to_pd` (`K=expm(S)` via `eigh`), `unpack_primitive`, `PrimitiveSpec.total_width`, `metric_GK_diag(_full)`. The env impedance branch (`act2impedance_tau`) is M7.
**Gate (done, fedguide).** `svec2sym(sym2svec(M))==M` for d∈{3,6}; `K=expm(...)` SPD + `jax.grad` finite; `action_size` layout (`test_mdac_primitive.py`, 9 pass).

### 4.2 Soft-feasibility (augmented-Lagrangian) weighting
**Math.** `J̃_k=J_cl+λ_k^T r+(ρ_k/2)‖r‖²` (eq:soft_feasibility_cost); `r=[h ; [g]_+]` (eq:constraint_residual); `log w_{k,m}=−(1/β_k)J̃_k+λ_{ψ,k}log p_ψ` (eq:importance_weight); softmax weights; `β=temp·rews.std()`. **Sign trap:** `J_cl=−rews.mean(-1)` (DIAL maximizes reward, AL is a cost). `λ_k`,`ρ_k` annealed across reverse steps (early lenient, late strict).
**Reuse (cfsmbd/mdcoas — NOT mdoc).** the cfsmbd AL pattern `rollout_augmented_reward_and_v` (L364), `aug_lambda/aug_rho`. **New (done):** `env_rollout.build_brax_rollout_augmented(env, aug_lambda, aug_rho)` folds `−(λ·mean[g]₊ + ρ/2·mean[g]₊²)` into the brax reward, reading `[g]_+` from the env `constraint_residual` hook (default no-op ⇒ 0 penalty ⇒ byte-identical plain rollout). `aug_lambda/aug_rho` are MDAC config. (Separate from the `constraint_filter` seam, which is mdoc's action-space filter.) `ALMAdaptiveScheduler` (adaptive λ/ρ) is optional/future.
**Gate (done, docker).** AL-off == DIAL **byte-identical** on real brax (0.00e+00); AL-on drives a synthetic violation 2.556→0.000; closed-loop 1.997→0.000 (`test_mdac_al_docker.py`, `test_mdac_m5_docker.py`).

### 4.3 Constraint-manifold geometry: metric `G_k` + tangent projection `P_M^G` + retraction
**Math.** `G_k=I+ρ_k J_C^T W_C J_C+κ_k J_tr^T W_tr J_tr+G_K` (eq:method_metric); `Ŝ_{M,k}=P_M^{G_k}(U_k)·G_k^{-1}·Ŝ_k` (eq:geometry_shaped_score); `P_M^G=I−G^{-1}J_C^T(J_C G^{-1}J_C^T)^† J_C` (eq:metric_tangent_projection); `ε_{M,k}=P_M^{G_k}ε_k` (eq:tangent_noise); retraction QP `argmin ½‖U−U^raw‖²_{G_k} s.t. C=0, K=exp(S)` solved by 1-2 linearized steps (eq:linearized_retraction). **`J_C` is evaluated on clean predicted states** (`jac_mode='clean_state'`): differentiate only through the constant spline matrix + analytic kinematics/contact Jacobian — **never through the mjx rollout**.
**Reuse (genemetry — NOT hand-rolled).** `manifold/sdf.SdfManifold.geometry(a_geom, topk, eps)` (build active rows + metric_sys/tan_sys) + `.project(v, bundle, mode='metric'|'tangent')` (`ops/jax_ops.project_complement_batch`, Cholesky/Woodbury); `retraction/cfs.CfsRetraction.retract`; `schedule/overlay.ScheduleOverlay`. 2GO `twogo_jax.py` is the working analog (dense space, per-step vmap).
**New (done):** the geometry seam in `mdac_jax.py` — when the env/solver supplies `geometry_fn(state, Ybar_nodes, t0) -> a_geom (Hnode+1, nu)` (analogous to 2GO `_constraint_geometry_time_jit`), the backend calls `SdfManifold.geometry/project` on the update direction (+ optional `CfsRetraction`). No `geometry_fn` ⇒ skipped ⇒ DIAL. `mdac_topk_active`(clamped ≤Hnode+1)/`mdac_eps_stab`/`mdac_geom_gain` config.
**Gate (done, docker).** geometry-on runs on real brax (jit+vmap+scan over brax States), differs from geometry-off (max|Δ|=0.687), and suppresses the constrained direction (mean|u[:,0]| 0.431→0.123) — `test_mdac_geometry_docker.py`. **TODO (M7):** spatial-obstacle `geometry_fn` (SDF grad→control, clean-state Jacobian, no mjx backprop) + `CfsRetraction` filter_fn for corridor/stepping/contact.

### 4.4 Model-free / learned priors — **SHARED infra** (`genedynamics/learning/priors/`), consumed by MDAC

> **This is NOT an MDAC-private component.** The RL prior `p_ψ`, the learned diffusion prior `s_θ`, and the elite buffer are general — any solver (mbd / cfsmbd / twogo / mppi / future) can consume them through one `prior=` seam. So they are built as a new top-level package and MDAC is simply the first heavy consumer. (Hoisted out of the old `mdac/priors/` nesting.)

**Math (MDAC's usage).** Warm-start `U^init=λ_shift·U^shift+(1−λ_shift)·U^rl` (eq:rl_warm_start); log-prior term `+λ_{ψ,k}·log p_ψ` in each candidate log-weight (eq:importance_weight); prior factorizes `p_ψ(U)=Π_h π_ψ(u_h|ô_h,c)` (eq:rl_prior_horizon). Optional `s_θ`: elite optimized-`U` buffer → conditioned diffusion → fused score `Ŝ_k=ω^mb Ŝ^mb+ω^mf s_θ` (eq:joint_score; minimal impl `ω^mf=0`). Train PPO (humanoid) / SAC (manipulator).

**IMPLEMENTED — `genedynamics/learning/priors/` (multi-backend, like the solvers).** The RL prior keeps the multi-backend architecture; the **jax backend integrates brax's training** (decision 2026-06-22).
```
genedynamics/learning/priors/
  base.py            # Prior protocol (output_dim · act · logp_of_sequence · warm_start);
                     # DiffusionPrior(Prior): + score(x,t) -> transport score_g
  registry.py        # register_prior / make_prior("rl"|"diffusion")  (codesign reserved)
  elite_buffer.py    # EliteBuffer: add / topk / sample(elite_frac) -> s_theta data
  rl/                # multi-backend RL policy prior
    rl_prior.py      #   RLPrior orchestrator (backend dispatch, like solvers)
    backend_impl.py  #   RLPriorBackend Protocol
    backends/brax_jax.py   # BraxRLPrior: brax PPO networks + params; act via make_inference_fn;
                           #   logp_of_sequence = sum_t dist.log_prob(logits, inverse_postprocess(a));
                           #   warm_start = policy mean tiled to (n_warm_nodes, A). brax lazy-imported.
  diffusion/         # multi-backend learned s_theta
    diffusion_prior.py     # LearnedDiffusionPrior orchestrator
    backends/jax.py        # JaxDiffusionPrior: DDPM score-MLP; score(x,t)=-eps/sqrt(1-abar);
                           #   train(elite data); warm_start via ancestral sampling. Pure jax.
genedynamics/learning/train_rl_prior.py   # brax PPO train -> (params, config); build_rl_prior(...)
```
**Reuse:** `core/prob/noise_sampler` (the additive-seam pattern the `prior=` seam mirrors); `transport.step` `score_g` (conduit for `s_θ.score`); brax `make_ppo_networks`/`make_inference_fn`/`NormalTanhDistribution` (jax RL backend). **Deferred:** lift `mrmfmbd/theta_prior.ThetaPrior`→`CoDesignPrior` (touches mrmfmbd; back-compat shim); the `λ_ψ·log p_ψ` log-weight term (needs obs-from-rollout plumbing) — only the **warm-start mix** is wired so far.

**The seam (MDAC, additive — `None ⇒ byte-identical DIAL).** `MDACSolver(prior=, prior_lambda_shift=)`; `MdacBackendJax.replan` mixes `U_init = λ_shift·U_shift + (1−λ_shift)·U_rl` (eq:rl_warm_start) when a prior is present. Kept MDAC-local (does NOT touch the shared `base_solver`, honoring "don't touch the MBD algorithm").

**Gate (done).** fedguide (`test_mdac_priors.py`): registry, EliteBuffer, jax diffusion `score`/`train`/`warm_start` + protocol conformance, prior-seam additive safety (`prior=None` Δ=0; `λ_shift=1` identity; `λ_shift=0` changes the plan). docker (`test_mdac_prior_docker.py`): brax tiny-PPO → `RLPrior` → `warm_start (Hnode+1, A)` + `act (A,)` finite; MDAC-with-prior ≠ MDAC-without on real brax. **`s_θ` training-from-elites is scaffolded; the full PPO/SAC sweep + `ω_mf>0` path is follow-up.**

### 4.5 Generalized reverse (DDPM/DDIM/FM) + coupled annealing
**Math.** Clean proposal `Ũ~N(U_k/√ᾱ_k,(1/ᾱ_k−1)Σ_k)` (eq:method_clean_proposal — the `√ᾱ` rescale DIAL lacks). DDPM raw update `U_{k-1}^raw=(1/√α_k)(U_k+(1−ᾱ_k)Ŝ_{M,k})+σ_k ε_{M,k}` (eq:ddpm_control_update); DDIM/flow via predicted clean `Û_{1,k}` (eq:ddim_control_update); `σ_k>0` diffusion, `σ_k=0` flow. Coupled schedule `(σ_k↓, ρ_k↑, κ_k↑, λ_{ψ,k}↓)` (eq:coupled_schedule, eq:schedule_trend), `ᾱ_k∈(0,1]↑`, `1-ᾱ` clipped from 0.
**Reuse (NOT new).** `transport=` seam → `AdaptiveTransport` (+ `ddpm_jax/ddim_jax/fm_jax`); call `transport.step(tau_k,tau1_k,eps_k,score_g,sched={abar_k,alpha_k,abar_km1})` (mdoc_jax.py:383 pattern); `None` ⇒ verbatim weighted mean (DIAL). Schedule via genemetry `ScheduleOverlay.compute(margin,rho,eta_base)`; DIAL `make_sigma_control`/`make_traj_diffuse_factors` for the sampling noise. **No `mdac/core/schedule.py`, no `strategy_base.py`** — those were deleted.
**Gate.** `transport=None` ⇒ byte-identical DIAL (regression); `DDPM(σ=0)==DDIM(η=0)` is the upstream transport's own gate, not MDAC's to re-prove.

---

## 5. Milestone plan

Current state: **DIAL done, bridge done, env_rollout done, node-spline done.** **M0-M4 DONE in the corrected reuse architecture (2026-06-22):** MDAC = a thin compose-solver in the mdoc/twogo high-perf pattern (`mdac_jax.py` = jit + lax.scan reverse-diffuse, vmap rollout) that reuses the upstream seams (§3.3 table); the only-new piece (SPD primitive) is upstream in `core/control/stiffness.py`. **15 mdac tests + 29 dial/bridge regression pass; seams-off == DialBackendJax byte-identical (≤1e-5).** Track A = hardware-free (fedguide x86). Track B = native arm64 / GPU (mjx). **DIAL byte-golden is a BLOCKING regression at every milestone.**

| M | Goal | Status |
|---|---|---|
| **M0** | Scaffold `solvers/single/mdac/`; `register_solver("mdac")`; DIAL-default config; `method_registry` (MethodFlags + single-flag ablations + `assert_fair`); `MdacBackendJax` = jit+vmap+scan kernel; native `WarmStartPlanner`. | **DONE** — seams-off `plan()` ≤1e-5 byte-identical to `DialBackendJax`. |
| **M1** | §4.1 SPD primitive **upstream** `core/control/stiffness.py` (not solver-private). | **DONE** — svec round-trip; `K=expm` SPD + grad finite; `action_size` regression (9 tests). |
| **M2** | §4.5 reverse via the `transport=` seam (`AdaptiveTransport` reuse), schedule via `ScheduleOverlay`. | **DONE (wired)** — `transport=None`⇒DIAL; DDPM seam routes + runs. Full DDIM/FM annealing tuning = on constrained task (Track B). |
| **M3** | §4.2 soft-feasibility via the `constraint_filter=` seam (`core/constraints`, mdoc pattern). | **DONE (wired)** — NoOp⇒DIAL byte-identical; filter seam invoked + changes result. *Brax-reward AL (cfsmbd pattern) via `env.constraint_residual` = TODO for constrained brax tasks.* |
| **M4** | §4.3 geometry via the genemetry seam (`SdfManifold.geometry/project` + `CfsRetraction`, 2go pattern), gated on env `geometry_fn`. | **DONE (wired)** — no `geometry_fn`⇒skipped⇒DIAL; with injected manifold the projection routes through genemetry (CPU smoke). Constrained-env e2e = Track B. |
| **M5** | Run the bridge-wired `solve()` on a **constrained** task (corridor/stepping/contact) + golden freeze; verify manifold-active ≠ DIAL there, byte-identical DIAL on unconstrained. | **bridge wired**; needs native arm64/mjx + a constrained env with a `geometry_fn` / AL residual. |
| **M6** | §4.4 **shared** `genedynamics/learning/priors/` pkg (Prior/DiffusionPrior/RLPrior/EliteBuffer/registry) + `prior=` seam on the base solver (additive, all solvers); lift mrmfmbd `theta_prior`→`CoDesignPrior`; wire MDAC warm-start mix + `λ_ψ`; then PPO/SAC train + elite buffer + `s_θ`. | A (pkg + logp math + additive-safety regression) + B (train) | `prior=None` bitwise-equal M5 **for every seamed solver**; `jit` compiles with prior on; warm-start endpoints; `make_prior` round-trips; `CoDesignPrior`==old mrmfmbd; `s_θ` after elite collection. |
| **M7** | Exp envs (§6): humanoid box-push SE(2) + arm surface-scan; `surface_geometry.py` (NURBS). | B (envs) + A (NURBS math) | face one-hot→contact point; residual sign tests; `J` monotone toward goal; `mppi` 50 steps moves toward goal, fall_rate=0, no NaN. |
| **M8** | Baseline/ablation harness (`method_registry.py`) + eval (metrics, CVaR95, violation rates, RQ1-5 sweep, bootstrap CI, LaTeX tables). | A (registry/metrics) + B (runs) | fairness self-check (same `(M,H,K)`); 8 ablations each bypass exactly one component; metrics unit tests; aggregate emits table + plots. |

**Critical path:** `M0 → {M1 ∥ M2 ∥ M3 ∥ M4} → M5 → M6 → M7 → M8`. M1-M4 are Track-A-heavy and parallelizable (≈45% of load-bearing code is hardware-free). M6's `s_θ` strictly follows minimal-prior elite collection. M8 last (needs registered MDAC + envs + RL baselines).

---

## 6. Experiments & baselines

MDAC differs from DIAL/MBD/2GO/cfsmbd **only on constrained, contact-rich tasks** (§2.6) — so every experiment is constrained. Experiments are NOT bespoke scripts: the repo has a **plugin + runner framework** that wires `env × method × metrics × obstacles × seeds × levels` and aggregates. Build MDAC's experiments AS plugins on it.

### 6.1 The framework (architecture map)
| Piece | File:line | Role |
|---|---|---|
| Plugin ABCs | `experiments/framework/base.py:19-273` | `MethodPlugin` (`create_planner(env,energy,cfg)`, `plan(planner,x0,rng)→{states,actions,candidate_*}`, `name`); `EnvironmentPlugin` (`create_env`/`create_energy`/`get_state_dim`/`extract_position`/`name`); `MetricsPlugin` (`compute(traj,env,obstacles,constraints,**kw)→dict`, `name`); `ObstacleGeneratorPlugin` (`generate(level,seed,start,target,cfg)`); `VisualizationPlugin`; `BaselineProtocol` |
| Registry | `experiments/framework/registry.py::PluginRegistry:12-117` | 5 plugin kinds: method / environment / metric / visualization / obstacle_generator |
| Runner | `experiments/framework/experiment.py::ExperimentRunner:78-711` | `run_all()` (L683) = `for level: for seed: run_single_experiment` → env→start/target→obstacles→energy→constraints→scheduler→`create_planner`→`plan`→best-candidate select→Trajectory→metrics→viz→save |
| Config | `experiments/framework/config.py::ExperimentConfig:22-183` | YAML: `name, output_dir, env_name, env_params, method, method_params, obstacle_levels, obstacle_config, seeds, backend, metrics, visualizations, constraint_config, scheduler_config` |
| Central registration | `experiments/runner.py::register_all_plugins:191-266` | where every plugin is registered; **run** `python -m genedynamics.experiments.runner <config.yaml>` |
| Output | `results/<name>/` | `experiments/level_*_seed_*.json` (traj+metrics) · `summary.json` (per-level mean/std/min/max + CVaR95) · `report.html` |

**Already-registered methods** (`runner.py:199-212`): `mbd, mbd3d, mdoc, ebmbd, cfsmbd, cfsmbd_full, 2go, mrmfmbd, d3il_unified` (+ `dpcc/safediffuser` conditional). **`mppi`/`cem`/`dial` solvers exist but are NOT yet MethodPlugins** (§6.3). Existing metrics: `ssr, obstacle_density, nonconvexity, episode_outcome, stepping_stones, corridor` (`runner.py:240-245`). Existing constrained envs/generators: corridor, stepping-stones (2D), D3IL avoiding — the **fast iteration substrates** before the contact-rich `idea.txt` tasks.

### 6.2 Recipe — add a new task (EnvironmentPlugin)
1. **Env class** — implement the brax/flat env (reward, constraints, primitive width via `action_size`). Constrained envs must expose the SDF/manifold residual the solver reads (`base_env.constraint_residual`, §4.2).
2. **Plugin** — `experiments/plugins/environments/<task>.py`: subclass `EnvironmentPlugin`; `name` (== `env_name` in yaml), `create_env(cfg)` (→ `make_env(self.name, **cfg)` or direct), `create_energy()`, `get_state_dim()`, `extract_position(state)`. Template: `plugins/environments/single_integrator_2d.py:12-69`.
3. **Register** — import in `plugins/environments/__init__.py` + `runner.register_plugin(<Plugin>(), 'environment')` at `runner.py:215-237`.
4. **Obstacles** — reuse `box2d/box3d` (set `obstacle_config.generator`), or new `ObstacleGeneratorPlugin` (`plugins/obstacles/box2d.py:17-72`). For contact tasks the "obstacle" is the constraint manifold (surface / box / contact set).
5. **Metrics** — reuse, or new `MetricsPlugin` (§6.4).
6. **Config + run** — write `configs/<task>.yaml`; `python -m genedynamics.experiments.runner configs/<task>.yaml`.

### 6.3 Recipe — add a baseline (solver → MethodPlugin)
1. **Plugin** — `experiments/plugins/methods/<m>.py`: subclass `MethodPlugin`; `create_planner(env,energy,cfg)` builds the solver from `cfg` (Nsample/Ndiffuse/horizon/...); `plan(planner,x0,rng)` returns the normalized `{states,actions,candidate_*}` dict. Template: `plugins/methods/mbd.py:16-100`, `twogo.py:20-115`.
2. **Receding/MPC** — for closed-loop, wrap with the shared bridge in `plan()`: `SolverPlannerAdapter(solver, horizon, warm_start_kwarg=...)` → `RecedingHorizonController(adapter, step_fn=dyn.step, n_steps, n_diffuse_init, n_diffuse).run(x0,rng)` (`solvers/common/receding_horizon.py:196-276`). **This is how every sampling baseline gets the same MPC treatment with zero per-baseline plumbing** — DIAL native, the rest via the adapter.
3. **Register** — `runner.py:199-212`.
4. **Fairness self-check** — same env+energy (all via the EnvironmentPlugin), same `horizon`, same sample budget `(M,H,K)`, accept `rng_key`. Enforce with `mdac/core/method_registry.py::resolve_method` (same `(M,H,K)` across methods; ablations only toggle one flag).

### 6.4 MDAC's experiments mapped onto the framework
- **Exp I — Surface-contact manipulation** (`idea.txt` 670-782). **NEW** `ArmSurfaceScanPlugin` (`plugins/environments/arm_surface_scan.py`) + `ArmSurfaceScanEnv` + `SurfaceCoverageEnergy`. 7-DoF arm; primitive `u^arm=(Δξ,Δη,Δψ,S_h,F^d_n)`; impedance law; constraints surface-attach `h_surf`, normal-align `h_normal`, force bounds `g_force`; surfaces S1–S4 (planar→unseen NURBS via `surface_geometry.py`).
- **Exp II — Humanoid box pushing** (`idea.txt` 784-954). Try to **reuse** `plugins/environments/humanoid.py` (HumanoidMjx) with `env_params.task_type=box_push` + the vendored `humanoid_h1_push_crate` scene (§2.4 — already has the box body); else NEW `HumanoidBoxPushPlugin`. Hybrid manifold `M_hum` (box-ground / hand-box face-select / stance-foot); `g_bal,g_fric,g_tip`; levels H1–H4 (double-support → walk-and-push → unjamming with face selection).
- **Fast substrates** before contact tasks: the existing constrained `corridor` / `stepping_stones` 2D envs — run MDAC vs DIAL/2GO there first to validate the manifold path (cheap, Track-A-adjacent).

### 6.5 Baselines (`idea.txt` 961-980) — status on this framework
| Baseline | Status | Build |
|---|---|---|
| MPPI, DIAL-MPC, CEM | solver exists, **no MethodPlugin** | thin `plugins/methods/{mppi,dial,cem}.py` (§6.3); DIAL native bridge, MPPI/CEM via `SolverPlannerAdapter` |
| 2GO, cfsmbd, mdoc, ebmbd | **registered** | reuse as-is (geometry/constraint baselines) |
| PPO, SAC (± learned stiffness) | **new** | `plugins/methods/{ppo,sac}.py` wrapping the §4.4 `RLPrior` policy as a standalone controller |
| SRL-VIC, ATACOM-RL, PegasusFlow/WBFO, RL+CBF/ISSA | **new** | one MethodPlugin each (`plugins/methods/`) |

MDAC is the only method with all of {MF prior, MB rollout, manifold safety}. **Fairness:** all sampling methods optimize the same `U`, same `F`, same `(M,H,K)`; DIAL/MPPI ablate tangent-projection+retraction but keep the sample budget.

### 6.6 Ablations & metrics
- **Ablations** (`idea.txt` 993-1011): one config per bypassed component, each toggling exactly ONE `MethodFlags` field in `method_params` (read in `create_planner`): w/o stiffness · fixed stiffness · w/o RL prior · w/o MB rollout · w/o tangent projection · w/o retraction · w/o adaptive schedule · Euclidean (vs log-SPD) stiffness. The `method_registry` fairness self-check guarantees only the toggled flag differs.
- **Metrics** — NEW `MDACMetricsPlugin` (`plugins/metrics/mdac_metrics.py`) returning, per run: **task success** (goal within margin) AND **constraint/contact/force/balance violation** as both **rate** and **CVaR95** (force-CVaR95, friction-cone, tip/balance margin, fall rate, contact loss/slip, coverage, stiffness smoothness, energy, runtime). Reuse `corridor._cvar` for CVaR95; the runner already aggregates mean/std/min/max + CVaR per level. The violation/CVaR columns are where MDAC beats sequential safety filters (RQ3) and fixed stiffness (RQ2) — report them alongside success, never success alone.
- **Worked end-to-end reference** (the smallest full loop to copy): `single_integrator_box_2d` + `mbd` — env plugin `plugins/environments/single_integrator_2d.py`, method `plugins/methods/mbd.py`, obstacles `plugins/obstacles/box2d.py`, a `configs/*.yaml`, `python -m genedynamics.experiments.runner <cfg>` → `results/.../{experiments,summary.json,report.html}`.

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
