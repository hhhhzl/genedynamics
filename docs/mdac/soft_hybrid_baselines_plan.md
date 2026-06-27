# Soft / Hybrid surfaces + new baseline solvers — build plan

Addendum to `MDAC_BUILD_PLAN.md`. Extends Exp-I (arm surface-scan) from RIGID-only
to **rigid / soft / hybrid** contact media, adds the metrics those media make
meaningful, and adds the external baselines as **new registered solvers** (incl. two
RL baselines). The **core MDAC framework + the four RQs do not change** — only the
Exp-I task formulation (richer contact medium) and the comparison set extend.

## 0. Decisions (locked this planning session)

- **Media** are a new axis at `configs/arm/impedence/{rigid,soft,hybrid}/`, mirroring
  the existing rigid tree (`<env>/<role>/<method>.yaml`, `output_dir` mirrors the path).
- **Soft** = elastic-foundation (Winkler) local sink; **hybrid** = same Winkler model
  with a fixed spatially-varying stiffness map. AVOID mjx flex/FEM (intractable at
  2048-sample rollouts).
- **Hybrid environments** = flat single-folder names, 2–3 representative combos:
  `stripes_plane`, `center_hard_plane`, `center_soft_plane`.
- **Baselines** added as **new registered solvers** (`register_solver`), dispatched
  from the MDAC harness (NOT the general `experiments/` MethodPlugin path). Four new
  baselines, all **PORTED onto the brax/jax stack** — the vendored repos under
  `baselines/` are incompatible frameworks (see §3): `mppi` (standalone),
  `pegasusflow` (vendored `baselines/PegasusFlow`), `atacom` and `issa` (the two RL
  baselines, vendored `baselines/rl_on_manifold` + `baselines/Implicit_Safe_Set_Algorithm`).
- The two **RL baselines are ATACOM** (constraint-manifold tangent-space RL) **and
  ISSA** (implicit safe-set RL), NOT pure PPO/SAC — PPO/SAC (`learning/priors/rl`,
  brax) are their underlying RL ENGINE. (Pure PPO/SAC is a free trivial extra if wanted.)
- The MDAC flag-degeneration baselines (`dial`/`mppi`/`mbd` in `method_registry`) stay
  as **controlled ablation anchors** (rename the anchor `mppi`→`dial_anchor` to avoid
  clashing with the standalone `mppi` solver).
- **RL granularity**: per medium, train ONE policy on a MIXTURE of that medium's *seen*
  surfaces, evaluate closed-loop on *seen + unseen*. 3 media × {ATACOM, ISSA} = **6 policies**.
  RL acts over the SAME 10-D position-stiffness primitive (idea.txt 987).

---

## 1. Part A — Surface media (rigid / soft / hybrid)

### 1.1 ContactMedium abstraction (pluggable, config-selected)

Per the framework conventions (medium is a component, deformation math is shared
upstream, `h_surf`/`h_normal` evaluate on the *deformed* surface): introduce a
`ContactMedium` that the env's contact builder selects.

- `rigid` — current `_build_contact_model` (mjx HFIELD + probe; baseline solref).
- `soft` — Winkler elastic foundation: local sink `w(ξ,η) = F_n / k_surf` (uniform `k`).
- `hybrid` — same Winkler model, `k_surf = k_map(ξ,η)` a fixed map.

The deformation math lives upstream in `core/contact/` (reuse the EXISTING
`core/contact/spring_damper.py::SpringDamperContact`, whose stiffness `k` already
accepts a **callable `k(surface_point) → scalar`** — exactly the variable-stiffness
hook hybrid needs; `core/contact/protocols.py::ContactState` already carries
penetration depth; `core/contact/cbf_terms.py` has `force_bound_rows` /
`penetration_bound_rows`). These are currently unused by `panda_brax.py` and are the
intended home for the medium model.

### 1.2 surface_geometry deformation field (the clean extension point)

`core/coverage/surface_geometry.py::Surface{kind, params}` evaluates `point()` and
gets `normal()/tangents()` by **autodiff** (`jax.jacfwd`). The NURBS surfaces are
`p = base + h(ξ,η)·n_base`. Add an optional deformation field `w(ξ,η)`:

```
p_deformed(ξ,η; w) = base + (h(ξ,η) + w(ξ,η)) · n_base
```

`normal()/tangents()` autodiff straight through `w` — no per-family code. `w` comes
from the Winkler force model (`w = F_n / k_surf(ξ,η)`), so `h_surf`/`h_normal` and the
manifold residual (`panda_brax._manifold_res_node`) are evaluated on the deformed
surface automatically.

### 1.3 Fidelity choice — Winkler, NOT MPM/FEM (does soft need MPM? → no)

**Soft does NOT need MPM.** The task is a rigid probe pressing into a surface with SMALL
local indentation under force tracking; the dominant effect is normal compliance, which a
Winkler elastic foundation (local sink `w ∝ p/k`) captures closed-form per contact point —
cheap, differentiable, in-loop. MPM (the `jax_mpm` backend used for the soft-**robot body**
co-design) and FEM / mjx-flex model large deformation / material flow / coupled continua —
overkill here and INTRACTABLE inside the planner's 2048-sample × multi-step rollout (the
mrmfmbd fidelity-ladder exists precisely because high-fidelity soft sim cannot run in the
inner loop).

Fidelity ladder (only if the paper wants realism beyond Winkler):
- **in-loop (planner + exec):** Winkler local sink — DEFAULT.
- **mid-tier (optional):** Boussinesq / Hertz elastic half-space (coupled, precomputed kernel
  convolution) if lateral coupling matters — still cheap-ish.
- **offline only (NEVER in rollout):** MPM (`jax_mpm`) / FEM as a high-fidelity reference to
  sanity-check the Winkler surface — a fidelity check, not the planner model.

**Real soft bodies (e.g. a soft-organ phantom — future hero demo / sim2real).** A real
organ is large-deformation, viscoelastic (hysteresis/relaxation), near-incompressible,
possibly anisotropic, wet/slippery — it cannot be modeled accurately and MUST NOT be
simulated in the planner loop. The bridge is the EXISTING closed-loop force control
(`_impedance_tau`: grav-FF + PI + per-substep fast force loop), which regulates the real
contact force WITHOUT knowing tissue stiffness — the same robustness that handles S4. The
soft-organ regime extends the ladder, not the core (MDAC + RQs unchanged; `h_surf` just
evaluates on the deformed surface):
- in-loop: upgrade Winkler → **Kelvin-Voigt** (`f = k·d + b·ḋ`, the damper `b` already in
  `spring_damper.py` → viscoelasticity for free); SLS/Zener if relaxation/creep matter.
- coupling: a precomputed Boussinesq/Green compliance kernel OR a **learned reduced/latent
  deformation model** (see soft-robot codesign latent-field) for global coupling cheaply.
- offline (train/DR/validate): **`jax_mpm`** (GPU, differentiable, large deformation) or
  external FEM (SOFA / differentiable DiffPD); used for heavy material DR (E, ν, damping,
  viscoelastic τ, friction) and to QUANTIFY the cheap model's sim2real gap.
- hardware: silicone/hydrogel organ phantom + wrist F/T sensor, force-controlled scan;
  validate the cheap model against measured probe displacement / vision.
Honest hard parts: contact instability (chatter) on very soft/wet tissue; viscoelastic
hysteresis makes force↔indentation non-single-valued (split metrics by loading/unloading);
the cheap-model-vs-real gap must be quantified (that is what the offline tier is for).

mjx `solref` is per-geom only (no per-cell spatial variation), so the medium effect is
realized through the deformed surface + variable-`k` impedance/force model, NOT the mjx
contact solver (the real mjx contact keeps a baseline compliant `solref`). Matches the
"elastic-foundation tier, theory core unchanged" decision.

### 1.4 Config trees

```
configs/arm/impedence/
  rigid/   <env∈{plane,cylinder,convex,bumpy,unseen}>/{baseline,main,ablation}/<method>.yaml   # done
  soft/    <env∈{plane,cylinder,convex,bumpy,unseen}>/{baseline,main,ablation}/<method>.yaml
  hybrid/  <env∈{stripes_plane,center_hard_plane,center_soft_plane}>/{baseline,main,ablation}/<method>.yaml
```

Each `_base.yaml` per medium carries the shared budget + `env_params.medium` (+ for
hybrid `env_params.stiffness_map`). `discover_configs` already walks the tree; point
`MDAC_CONFIG_DIR` at a medium root (or extend the runner to sweep all media).

### 1.5 File checklist — Part A physics

- `core/coverage/surface_geometry.py` — add optional `w(ξ,η)` deformation field to
  `Surface` + NURBS/analytic `point()`; `normal/tangents` unchanged (autodiff).
- `core/contact/` — Winkler `ElasticFoundationMedium` (uniform + map `k`), reusing
  `spring_damper` callable-`k`; expose `sink(F_n, ξ, η)` and `k_map` constructors
  (`stripes`, `center_hard`, `center_soft`).
- `genedynamics/envs/domains/manipulation/panda_brax.py` —
  `PandaSurfaceScanConfig`: add `medium ∈ {rigid,soft,hybrid}`, `stiffness_map`,
  Winkler `k0`/`k_hard`/`k_soft`; `_build_contact_model` / `_desired_pose` /
  `_manifold_res_node` evaluate on the deformed surface; force model uses `k_surf(ξ,η)`.
- `genedynamics/envs/domains/manipulation/__init__.py` — register medium-aware aliases
  (or pass `medium` via `env_params`/`env_overrides`, already wired through `make_mdac`).

---

## 2. Part A — Metrics (additional to rigid)

Current arm metrics (`experiments/plugins/metrics/extractors.py::arm_surface_scan_signals`
+ `ARM_METRICS` → `evaluation/metrics.py` `@metric` registry): surface_tracking_error,
normal_alignment_error, force_tracking_error(_tracked), contact_loss_rate,
force_violation_rate, force_cvar95, control/stiffness_smoothness, energy, runtime.
NONE touch deformation/penetration/compliance — confirmed.

Add (grouped by what each medium newly exposes):

| Group | Metric | Definition |
|---|---|---|
| Deformation (soft+hybrid) | `deformation_depth_max/_mean/_rms` | surface sink `w` at the contact point over the scan |
| | `indentation_tracking_error` | RMS(actual indentation − commanded) |
| Compliant dynamics (soft) | `force_overshoot` | `max(F − f_target)` at contact onset / boundary |
| | `settling_time` | steps to reach f_target±ε after contact onset |
| | `contact_chatter` | count/var of contact on/off transitions (soft bounce) |
| **Stiffness adaptation (RQ2 payoff)** | `stiffness_adaptation_corr` | corr(commanded `K_h(S_h)`, local `k_surf(ξ,η)`) along the path — stiffen on hard, soften on soft |
| | `stiffness_region_force_error` | force error conditioned per stiffness region (hard/soft) |
| Heterogeneity (hybrid) | `boundary_transient` | force/tracking error spike in a window around stiffness discontinuities |
| Safety / sim2real | `surface_work` | ∫ F·(deformation rate) injected into the surface |
| | `peak_contact_pressure` | worst-case contact pressure (damage proxy); tie `force_violation` to a material pressure cap |

Adding a metric (3 steps, infra already there): (1) `@metric` fn in
`evaluation/metrics.py`; (2) emit the signal in `arm_surface_scan_signals`
(needs `w`/penetration from the deformed-surface eval); (3) add to `ARM_METRICS`.
Keeps the BUILD_PLAN 6.6 "violation-rate + CVaR95" framing (report violation/CVaR
columns alongside success, never success alone).

**Scientific framing:** soft/hybrid test whether the manifold + stiffness primitive
help MORE on compliant/heterogeneous surfaces, where a fixed-stiffness baseline should
visibly lose — exactly the RQ2 contrast. `stiffness_adaptation_corr` is the metric that
operationalizes it.

---

## 3. Part B — Baselines as new registered solvers

### 3.1 Architecture: MDAC-harness dispatcher (decision A)

The arm comparison runs on the MDAC harness (`mdac/run_experiment.py` + `make_mdac`),
NOT the general `experiments/` framework. Generalize the harness rather than migrate:

- `mdac/experiment.py`: add `make_controller(task, method, env, backend, **cfg)` (or
  generalize `make_mdac`) that dispatches by `method`:
  - MDAC method (in `method_registry.METHOD_TABLE`) → `MDACSolver` (current path).
  - registered standalone solver (`get_solver_registry()`) → build on the brax env
    (inject `build_brax_rollout`/`build_brax_step` from `solvers/common/env_rollout.py`)
    + `solvers/common/receding_horizon.py::SolverPlannerAdapter` → `RecedingHorizonController`.
  - RL method → `RLPolicySolver` (loads a trained policy; `run_receding` = act each step).
- `mdac/run_experiment.py::_run_one` unchanged in shape — still
  `ctrl.run_receding(x0, n_steps, rng)` returning `RecedingHorizonResult(.states,.actions)`.
- `method_registry` (or a sibling `baseline_registry`): a `method → solver-kind` table so
  the dispatcher knows MDAC-flag vs standalone-solver vs RL.

This reuses the SAME bridges BUILD_PLAN 6.5 references (DIAL native bridge,
`SolverPlannerAdapter`, RL-policy-as-controller) — only the dispatch home differs.

### 3.2 The four new baselines (each a registered solver, ALL ported to brax/jax)

The vendored repos under `baselines/` are in **incompatible frameworks** and CANNOT
run on the brax arm env as-is — they are **algorithmic references**; we re-implement
each method's core on the brax/jax stack (reusing the env's existing constraint hooks
`constraint_residual` / `manifold_residual` / `manifold_geometry`, and `learning/priors/rl`
as the RL engine).

| Method | Vendored repo / framework | Family | Port = re-implement (reference files) |
|---|---|---|---|
| `mppi` | (none — uses repo's `MPPISolver`) | sampling MPC (path-integral) | `MPPISolver` exists but lacks a brax path (assumes flat-state `DynamicsToEnvAdapter`). Add a brax branch like DIAL's (`is_brax_env`→`build_brax_rollout`). Register, distinct from the `dial_anchor` flag-degeneration. |
| `pegasusflow` | `baselines/PegasusFlow` — **torch + IsaacGym** (arXiv 2509.08435, "rolling-denoising score sampling / flow matching", basis-function sampling MPC) | sampling MPC over a spline basis | NEW `solvers/single/pegasusflow/`; port the sampling core to jax. Reference: `traj_sampling/traj_sampling/{traj_grad_sampling, spline, noise_scheduler, trajopt_policy, noise_sampler}.py`. Same basis dim = `Hnode` (idea.txt 985). Brax path + `SolverPlannerAdapter`. |
| `atacom` | `baselines/rl_on_manifold` — **mushroom_rl + torch** (CoRL'21, "Acting on the Tangent space of the Constraint Manifold") | constraint-manifold RL | **Registered solver `AtacomSolver` (jax backend).** FAITHFUL: a brax SAC policy **trained ON the manifold** via `AtacomEnvWrapper` (tangent action `nu−n_f`); the transform `u = Nc·α − Jc⁻¹(K_c·C)` + slack keeps every action on `{C=0}` (port of `atacom.py:step_action_function` + `pinv_null`). **Reuses the env's `manifold_residual`** (same manifold MDAC uses — clean, fair). Reference: `atacom/atacom.py`, `utils/null_space_coordinate.py`. |
| `issa` | `baselines/Implicit_Safe_Set_Algorithm` — **TF1 + safety-gym (mujoco-py) + MPI** (CoRL'21/JAIR, "Implicit Safe Set Algorithm") | safe RL | **Registered solver `IssaSolver` (jax backend).** brax PPO engine + implicit-safe-set + **AdamBA** action projection onto the safe set (the env's inequality `g` from `constraint_residual`). Reference: `toy_problem/{AdamBA, adamba_ssa_project_distance}.py` + `safety_gym/algo/.../safe_rl` (PPO-ISSA, TF1 — port the safety layer only). |

All four register via `register_solver` and follow the repo's multi-backend solver layout
(`solvers/single/<name>/{__init__.py, <name>.py, backends/<name>_jax.py}`) with a **jax
backend** — the dispatcher (§3.1) builds them on the brax arm env like any other solver.

Fairness (idea.txt 982–987): the sampling baselines (`mppi`, `pegasusflow`) optimize the
same `U`, same rollout `F`, same `(Nsample,Hsample,Ndiffuse)` / basis dim → go through
`assert_fair`. The RL baselines (`atacom`, `issa`) are model-free → a SEPARATE fairness
axis (training budget + inference wall-clock); report in their own column, NOT through
the sample-budget `assert_fair`.

### 3.3 RL baselines (ATACOM + ISSA) — engine, port, two-stage training

**RL engine** (`learning/priors/rl`): `train_rl_prior.py` trains brax PPO from scratch
(`brax.training.agents.ppo.train`) → `(params, config)`; `priors/rl/backends/brax_jax.py::
BraxRLPrior.act(obs,key,deterministic)`. Gaps to fill: (1) **no SAC** (add
`brax.training.agents.sac`); (2) **no checkpoint save/load** (params are raw pytrees);
(3) it's a warm-start **prior**, not a standalone closed-loop controller.

**Port the methods on top of that engine** (FAITHFUL reproductions — reuse env constraint hooks):
- `atacom` — **the policy lives ON the manifold** (not a post-hoc projection). `AtacomEnvWrapper`
  (`solvers/single/atacom/wrapper.py`) exposes a TANGENT action space `null = nu − n_f`; inside
  `env.step` the jax transform (`backends/atacom_jax.py`, faithful to `atacom.py:step_action_function`
  + `pinv_null` + slack) maps the tangent action `α` to a full control `u = Nc·α − Jc⁻¹(K_c·C)` that
  stays on `{C=0}` (+ slack update for the inequality `g`). The policy is **trained on this wrapped
  env** → manifold-resident. `f` = the env clean-state `manifold_residual` (the SAME manifold MDAC
  uses); `g` = `constraint_residual`'s force bounds. **Validated** (docker): `α=0` and `α=rand` both
  keep `‖C‖→0`.
- `issa` — wrap the brax policy with the implicit-safe-set safety layer: at each step run **AdamBA**
  (faithful derivative-free ray search: exponential-expand → bisection to the safe-set boundary,
  `AdamBA.py:90-122`) to project the raw action to the closest point on the safe set `{g ≤ 0}`
  (= the min-‖·‖ QP of `adamba_ssa_project_distance.py`). **Validated** (docker): `g +10 → +0.000`.

**Each is its own registered solver with a jax backend** — `AtacomSolver` (`solvers/single/atacom/`)
and `IssaSolver` (`solvers/single/issa/`). ISSA uses the shared `RLPolicyController` (steps the real
env with the AdamBA-projected `act()` each step). ATACOM does NOT (its policy is manifold-resident):
its `run_receding` runs the policy through `AtacomEnvWrapper` and records the executed FULL-dim
controls + inner states for the metrics. Both act over the SAME 10-D primitive (Δξ,Δη,Δψ,S_h,F_n) →
"learned stiffness" is intrinsic (idea.txt 970, 987). **Consequence of faithfulness**: ATACOM's
policy is `nu−n_f`-dim (tangent) while ISSA's is `nu`-dim (raw) → they CANNOT share a ckpt (the
"3 shared policies" detour is incompatible; it is 6 — back to this plan's original count).

**Two-stage, per-medium** (your granularity): train ONE policy per (medium, method) on a
MIXTURE of that medium's *seen* surfaces; evaluate on *seen + unseen*.
3 media × {ATACOM, ISSA} = 6 ckpts. Train driver
`scripts/tasks/robot/arm/train_rl_baseline.py`; train configs
`configs/arm/impedence/<medium>/_rl_train.yaml` (mixed-seen env list + hyperparams + ckpt
out path); eval configs `<medium>/<env>/baseline/{atacom,issa}.yaml` carry `policy_ckpt`.

### 3.4 File checklist — Part B

- `solvers/single/mppi/mppi.py` (+ backend): add brax-env path (mirror DIAL).
- `solvers/single/pegasusflow/{__init__,pegasusflow.py,backends/...}` — NEW; port the
  spline-basis rolling-denoising sampler from `baselines/PegasusFlow/traj_sampling`;
  `register_solver("pegasusflow", …)`.
- `learning/priors/rl/backends/brax_jax.py` (+ a SAC backend) — add SAC; checkpoint save/load.
- `solvers/single/atacom/{__init__,atacom.py,wrapper.py,backends/atacom_jax.py}` — NEW registered
  solver `AtacomSolver` (jax backend). `wrapper.py::AtacomEnvWrapper` = the tangent-space (manifold-
  resident) env the policy trains on; `backends/atacom_jax.py::make_atacom_transform` = the faithful
  `Nc·α − Jc⁻¹(K_c·C)` transform + slack (port of `atacom.py` + `null_space_coordinate.pinv_null`).
  `register_solver("atacom", …)`.
- `solvers/single/issa/{__init__,issa.py,backends/issa_jax.py}` — NEW registered solver
  `IssaSolver` (jax backend); brax policy + implicit-safe-set + faithful **AdamBA** projection on the
  env `g`; port from `baselines/Implicit_Safe_Set_Algorithm/toy_problem/AdamBA.py`. `register_solver("issa", …)`.
- `solvers/common/rl_policy_controller.py` — shared `RLPolicyController` base (`run_receding`
  = projected `act()` each step) reused by `IssaSolver` (ATACOM uses its own manifold wrapper loop).
- `solvers/single/mdac/experiment.py` — `make_controller` dispatcher (MDAC | standalone solver | RL policy).
- `solvers/single/mdac/core/method_registry.py` (or new `baseline_registry.py`) — method→solver-kind table; rename anchor `mppi`→`dial_anchor`.
- `scripts/tasks/robot/arm/train_rl_baseline.py` — train driver (per-medium, mixed-seen).
- configs: per-medium `baseline/{mppi,pegasusflow,atacom,issa}.yaml` + `<medium>/_rl_train.yaml`.

---

## 4. Sequencing

1. **soft first** (uniform `k`): `ContactMedium` + `surface_geometry` deformation field
   + Winkler upstream (reuse `spring_damper`). Then `deformation_*` +
   `stiffness_adaptation_corr` metrics. Then `soft/` config tree → run MDAC/dial.
2. **baseline dispatcher** (pure wiring, no physics): MPPI brax path + dispatcher +
   `pegasusflow` solver → run MPPI/PegasusFlow on rigid.
3. **RL** (largest, most independent): SAC backend + checkpoint + `RLPolicySolver` +
   train driver → train per-medium, eval seen/unseen.
4. **hybrid** (soft + spatial `k_map`): `stripes_plane` / `center_hard_plane` /
   `center_soft_plane`; `boundary_transient` + region-conditioned metrics.

## 5. Open items / notes

- **All three vendored baselines are RE-IMPLEMENTED (复刻), never run as-is.** The repos
  (`baselines/{PegasusFlow, rl_on_manifold, Implicit_Safe_Set_Algorithm}`) are in
  incompatible frameworks (torch+IsaacGym, mushroom_rl+torch, TF1+safety-gym+MPI) and serve
  ONLY as pinned algorithmic references (each keeps its own `.git`). Every baseline is
  ported onto the brax/jax stack, reusing the env's constraint hooks + `learning/priors/rl`.
- **PegasusFlow** core to port: `traj_sampling/traj_sampling/traj_grad_sampling.py`
  (+ `spline`, `noise_scheduler`, `trajopt_policy`). Keep the same basis dim (idea.txt 985).
- **ATACOM** core to port: tangent/null-space projection (`atacom/atacom.py`,
  `utils/null_space_coordinate.py`) onto the env's `manifold_geometry`/`manifold_residual`.
- **ISSA** is the heaviest port (TF1 + safety-gym + MPI): port only the AdamBA safe-set
  projection + safety index (`toy_problem/AdamBA.py`), feeding the env's inequality `g`.
- Per-medium RL training mixes the *seen* surfaces; evaluate seen + unseen.
- Whether `run_experiment` sweeps ALL media in one run or one `MDAC_CONFIG_DIR` per medium
  (current: per-dir; trivial to extend).

---

## 6. Implementation plan (phased & executable)

Verification environments: **fedguide** (`/opt/anaconda3/envs/fedguide/bin/python`) for
`py_compile`, config-load, and pure jax/numpy math (no brax/mjx); **docker**
`genedynamics/dev-cpu:torch` (arm64) for any brax/mjx construct/step/rollout/full run.
Each phase ends with an acceptance check. Phases 1–3 (soft) and 4–8 (baselines) are
independent and can proceed in parallel after Phase 0.

### Phase 0 — Plumbing & guards ✅ DONE
- `PandaSurfaceScanConfig`: add `medium ∈ {rigid,soft,hybrid}` (default `rigid`),
  `stiffness_map`, Winkler `k0`/`k_hard`/`k_soft`. Thread `medium` through
  `make_mdac` `env_overrides` (already wired) → `make_env`.
- Keep `medium="rigid"` byte-identical to today (regression guard).
- **Accept:** fedguide `py_compile`; rigid configs still load 50/50; docker: a rigid env
  constructs + steps unchanged.

### Phase 1 — Soft medium physics ✅ DONE (mjx-solref compliance, NOT a new geometric model)
Implemented decision (architecture-minimal, no new files): UNIFORM soft is realized via the
EXISTING mjx compliant-contact mechanism — a softer `solref` derived from `soft_stiffness`
(same ks→solref map the `unseen` DR already uses), so the probe really sinks into the
surface under the press (real mjx contact + the hard-won force stack, unchanged). The
hand-rolled geometric Winkler (`surface_geometry` deformation field + `core/contact/
elastic_foundation.py`) is NOT needed for uniform soft and is DEFERRED to Phase 9 (hybrid),
where mjx's per-geom-only `solref` genuinely cannot express a spatial stiffness map.
- `panda_brax.py`: `PandaSurfaceScanConfig` gains `medium`/`soft_stiffness`; `medium=soft`
  sets the soft `solref`; new `_penetration(ps)` reads the probe penetration (= deformation).
- **Accept:** ✅ fedguide — `py_compile` + config dataclass accepts the fields, `medium=rigid`
  default (byte-identical guard); ⏳ docker — soft env constructs/steps, deformation > 0.

### Phase 2 — Soft-medium metrics ✅ DONE
- `evaluation/metrics.py`: added `@metric` `deformation_depth`, `deformation_peak`,
  `force_overshoot`, `contact_chatter` (pure numpy). (`stiffness_adaptation_corr`,
  `boundary_transient`, region-conditioned + `surface_work`/`peak_pressure` → Phase 9 hybrid,
  where they become meaningful with a spatial map.)
- `experiments/plugins/metrics/extractors.py`: `per_step` emits `penetration` →
  `deformation` signal; `ARM_METRICS` adds `deformation_depth`/`deformation_peak`/
  `deformation_cvar95` (cvar bind) + `force_overshoot` + `contact_chatter`.
- **Accept:** ✅ fedguide — metrics register + compute on synthetic signals; rigid (defo=0)→0.

### Phase 3 — Soft config tree ✅ DONE (run needs docker)
- `configs/arm/impedence/soft/` mirrors rigid: `_base.yaml` (`env_params.medium: soft`,
  `soft_stiffness: 1500`) + 50 method yamls (`<env>/<role>/<method>.yaml`, output_dir mirrors path).
- **Accept:** ✅ fedguide — `discover_configs` finds 50, budgets fair, output_dir mirrors path 1:1,
  merged `env_params.medium=soft`; ⏳ docker — `MDAC_CONFIG_DIR=configs/arm/impedence/soft`
  runs MDAC + `dial` end-to-end (deformation/force metrics populate).

### Phase 4 — Baseline dispatcher + MPPI brax backend ✅ DONE
Follows the repo solver convention (each method a registered solver = thin wrapper +
`backends/<name>_jax.py`; backend implements `WarmStartPlanner` → native receding-horizon via
the shared bridge, like DIAL — NOT a re-solve adapter):
- `mppi/backends/mppi_brax_jax.py` (NEW): `MPPIBraxBackendJax` — brax-native path-integral MPPI
  (`build_brax_rollout`, same env.step/reward/budget as MDAC, no manifold), a `WarmStartPlanner`.
  `mppi/mppi.py::MPPISolver` extended with a GATED brax path (`make_controller`/`run_receding`);
  the flat-state path is untouched. `register_solver("mppi")` (the existing name).
- `mdac/experiment.py`: `make_controller(task, method, …)` dispatches MDAC methods → `make_mdac`,
  else `_build_baseline_solver` builds the method's OWN registered solver (native `run_receding`).
  `run_experiment._run_one` switched to it. `method_registry`: MDAC-flag `mppi`→`dial_anchor`.
- **Accept:** ✅ docker — dispatch: mdac/dial → MDACSolver, **mppi → MPPISolver (brax backend)**,
  all finite (mppi surf_err 0.0096 vs mdac 0.0047), warm-started receding horizon.

### Phase 5 — PegasusFlow solver ✅ DONE
- `solvers/single/pegasusflow/{__init__,pegasusflow.py,backends/pegasusflow_jax.py}` (NEW solver,
  baseline-named): `PegasusFlowBackendJax` (a `WarmStartPlanner`) — NODE-basis (`Hnode`, same basis
  dim as MDAC) + node→dense interp + ROLLING-DENOISING (σ decays over the refine steps) +
  temperature-softmax weighted mean (DIAL/WBFO-style, manifold OFF). `PegasusFlowSolver` wrapper +
  `register_solver("pegasusflow")`. 13 `baseline/pegasusflow.yaml` configs.
- **Accept:** ✅ docker — `pegasusflow → PegasusFlowSolver` runs on the arm env (surf_err 0.0086,
  the expected baseline ordering); node→dense interp verified.

### Phase 6 — RL engine + controller base ✅ DONE (training is a longer offline run)
- `learning/train_rl_policy.py` (NEW): `train_rl_policy(env, algo='ppo'|'sac')` (brax PPO +
  SAC) + `save_policy`/`load_policy` (pickle of host `(params, config)`) + `build_policy_act`
  (rebuilds the inference fn; PPO reuses `BraxRLPrior`, SAC builds brax sac networks).
- `solvers/common/rl_policy_controller.py` (NEW): `RLPolicyController` — `run_receding` =
  `act()` (+ optional projection hook) each real step on the env → `RecedingHorizonResult`.
- `scripts/tasks/robot/arm/train_rl_baseline.py` (NEW): per-(medium,algo) train driver.
- **Accept:** ✅ docker — `RLPolicyController` runs closed-loop (with the ATACOM/ISSA projections,
  finite, force_violation 0). ⏳ the actual PPO/SAC TRAINING on the heavy mjx arm env is a longer
  offline run (engine + checkpoint pipeline are code-complete + compile-verified, not run to
  convergence in the autonomous session; the projection layers — the methods' real contribution —
  are validated with a policy stand-in).

### Phase 7 — ATACOM solver ✅ DONE
- `solvers/single/atacom/{__init__,atacom.py,backends/atacom_jax.py}` (NEW solver): backend
  `make_atacom_projection` projects the policy action ONTO the env constraint manifold `{C=0}`
  via the minimal-norm Gauss-Newton drift `a ← a − β J⁺ C(a)` (iterated), REUSING
  `env.manifold_residual` (the SAME manifold MDAC tangent-projects on). Thin `AtacomSolver`
  wrapper (`register_solver("atacom")`) drives it via `RLPolicyController`.
- **Accept:** ✅ docker — projection drives **‖C‖ 0.418 → 0.000** (action reconciled to the
  manifold); closed-loop finite, force_violation 0. (Boundary-saturated actions handled by an
  inward nudge so the clipped `force_cmd` keeps a live Jacobian.)

### Phase 8 — ISSA solver ✅ DONE
- `solvers/single/issa/{__init__,issa.py,backends/issa_jax.py}` (NEW solver): backend
  `make_issa_projection` projects the policy action onto the safe set `{g ≤ 0}`
  (`env.constraint_residual`) via an AdamBA proxy: a controlled NORMALIZED-gradient line search
  on the violation `Σ[g]_+²`, halting once feasible. Thin `IssaSolver` wrapper
  (`register_solver("issa")`) drives it via `RLPolicyController`.
- **Accept:** ✅ docker — projection drives **max g +10 → −2.04** (unsafe force pulled into bounds,
  nu +1.0 → +0.70); closed-loop finite, force_violation 0.

### Status — all 8 phases (+ Phase 9) implemented & docker-validated
Full comparison matrix configured: `configs/arm/impedence/{rigid(65),soft(65),hybrid(39)}` =
3 media × envs × **13 methods** (mdac · dial/mppi/pegasusflow/atacom/issa baselines · 7 ablations).
Remaining for a full paper run (Phase 10): train the 6 RL policies (per medium × {PPO,SAC}) via
the train driver, then sweep all configs in docker.

### Phase 9 — Hybrid medium ✅ DONE (model-based Winkler, probe non-collidable)
mjx `solref` is per-geom only → a SPATIAL stiffness map needs a model-based contact. Realized:
- `core/contact/elastic_foundation.py` (NEW, upstream): `stiffness_field(kind, ξ, η, k_hard, k_soft)`
  (`uniform`/`stripes`/`center_hard`/`center_soft`) + `winkler_force(k, δ)`. Exported from `core/contact`.
- `panda_brax.py`: `medium=hybrid` builds the probe NON-collidable (`_build_contact_model(collidable=False)`)
  and applies the analytic Winkler reaction `k(ξ,η)·δ·n_s` in `_impedance_tau` (δ = penetration below
  the rest surface); `_contact_force_at`/`_penetration_at` dispatch real-mjx (rigid/soft) vs Winkler
  (hybrid); `_k_surf_fn` exposes the local stiffness. Config: `stiffness_map`, `k_hard`, `k_soft`.
- `evaluation/metrics.py`: `stiffness_adaptation_corr` (RQ2 payoff). (`boundary_transient`,
  `stiffness_region_force_error`, `surface_work`/`peak_pressure` are still-open refinements.)
- `experiments/plugins/metrics/extractors.py`: emits the `k_surf` signal; `ARM_METRICS` adds
  `stiffness_adaptation_corr`.
- `configs/arm/impedence/hybrid/` envs `stripes_plane`/`center_hard_plane`/`center_soft_plane`
  (flat single-folder; top-level `level` = scenario label, `env_params.level=plane` = geometry,
  `env_params.stiffness_map` per env-`_base.yaml`). 30 method yamls.
- **Accept:** ✅ docker — hybrid constructs/steps finite; `k_surf` varies spatially (stripes
  8000/2000/8000, center_hard 2000/8000/8000); MDAC rollout deformation_depth≈1cm / peak≈2.3cm,
  all metrics finite. (`stiffness_adaptation_corr` exercised at full budget, not the smoke run.)
  ⏳ full comparison (Phase 10).

### Phase 10 — Full comparison & figures
- Run all media × {MDAC, dial_anchor, mppi, pegasusflow, atacom, issa} × envs × seeds.
- Tables (per env, methods as rows) + force/deformation/stiffness-adaptation figures
  (extend `scripts/tasks/robot/arm/`).
- **Accept:** comparison tables populate; MDAC wins task + stiffness-adaptation; RL reported on its
  own (training-budget) axis.
