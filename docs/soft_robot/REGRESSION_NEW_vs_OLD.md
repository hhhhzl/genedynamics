# MR-MF-MBD: NEW (HEAD) vs OLD (`dc7de15`) Regression — What Broke, What's Fixed, What's Pending

**Status:** investigation + partial fix in progress. Last updated: 2026-06-21.

## TL;DR

The commit **`dc7de15`** ("OLD") ran our method (MBD co-design) at **ours ≫ CEM/CMA-ES** and was
fast. The current **HEAD** ("NEW", a ~28-commit rewrite that added the learned controller, the
risk-sensitive objective, hard validity, MF/MB priors, and budgeted adaptive multi-fidelity)
**lost to CEM and got ~3.5× slower**. Since model-based diffusion is a provably better sampler than
CEM's Gaussian evolution, *ours losing to CEM is a regression in the rewrite, not an inherent
weakness*. The core MBD update (`Y0 = Ybar + σ·scale·ε`; reward-weighted-mean denoise; `sigma_max=0.15`)
is **identical** in both versions — the regression is in **what reward signal feeds the denoise** and
**how many fidelity levels are rolled per step**.

| benchmark | ours | CEM | ours/CEM |
|---|---|---|---|
| **OLD `dc7de15`** (4-mode μ=0.3–0.6, reproduced) | **11.66** | 4.42 | **2.64× ✅** |
| **NEW (pre-fix)** (6-mode μ=0.05–1.3, ours+sin) | 2.41 | 7.17 (risk) | 0.34× ❌ |

---

## 1. Setup of the two runs

| | OLD `dc7de15` | NEW (HEAD) |
|---|---|---|
| MBD backend file | `…/mrmfmbd/backends/mrmfmbd_mbd_jax.py` (30 KB) | same path (60 KB, doubled) |
| core update | `Y0=Ybar+σ·scale·ε`; `w=softmax((R−R̄)/(σ_R·T_k))`; `Ybar=Σ w·Y0 + τ·scale·ξ` | **identical** |
| `sigma_max` | 0.15 | 0.15 (**identical** — NOT the cause) |
| denoise reward `R_m` | `logsumexp_c(log p(c) + R[m,c]/T_mode)` — **reward-favoring soft-MAX** (`_s1_marginalize_jax`) | `−τ·logsumexp_c(log p(c) − R[m,c]/τ)` — **risk soft-MIN** (`risk_sensitive_marginalize_jax`, default) |
| top-K leaderboard | plain **mean reward** over modes | **risk ρ\*** |
| fidelity per step | **blocked ladder** — 1 level/step (coarse→fine blocks) | **enumerate-ALL** — rolls all 3 levels every step for V̂ |
| MF prior on x | none | `+0.2·log p_MF(x)` every step |
| validity I(z) | none | hard `−1e4` mask folded into the softmax |
| co-design env | crawling_ground, [4,3,4] **raw voxel** (48-d occ), n_grid64, K=100, M=32 | same |

Reproduction harness: OLD ran from an isolated git worktree `/workspace/gd_old` with
`PYTHONPATH=/workspace/gd_old` (overrides the editable install → loads the OLD backend, verified).

---

## 2. Root cause (verified by an old-vs-new diff)

### (B) ~3.5× slower — one dominant cause
**Enumerate-all-fidelity-levels.** `_rollout_all_levels` rolls all M candidates at **every** fidelity
level (30+100+200 env-steps) *every* diffusion step, to estimate V̂ for the adaptive selection — then
uses only the selected level. OLD's blocked ladder rolled **one** level/step.
- OLD env-steps/candidate = 21·30 + 31·100 + 48·200 = **13,330**
- NEW enumerate-all = 100·(30+100+200) = **33,000** (≈ 2.5×; with validity/marginalize overhead ≈ 3.5×)
- Note: NEW is even slower than *always running fine* (100·200 = 20,000). So **adaptive fidelity as
  implemented currently COSTS MORE than doing nothing** — it pays for all levels to use one. This makes
  contribution ② (budgeted adaptive fidelity) net-negative, not net-positive, as written.

### (A) ours loses to CEM — ranked
1. **Objective flip (highest confidence).** Search + leaderboard switched from reward-favoring
   soft-MAX (OLD) to risk soft-MIN (NEW default `regime_posterior_mode="risk_sensitive"`). The soft-min
   sits near worst-mode and **flattens/compresses the signal that drives the denoise softmax** — exactly
   the signal that makes MBD a better sampler than CEM. Both the diffusion update *and* survivor
   selection are pulled to a near-worst-case surrogate.
2. **Persistent MF-prior pull.** `+mf_prior_weight·log p_MF(x)` (default 0.2) every step drags morphology
   toward the prior mean, away from high performers. OLD had no prior term.
3. **Validity weight-collapse.** `validity_logmask` returns `−1e4` for invalid candidates; in the common
   mixed case (e.g. 1–2 of M=8 valid) the softmax collapses to those few → **ESS→1 → Ybar ≈ a single
   candidate**, destroying the multi-sample averaging that makes MBD > CEM. (All-invalid fallback only
   fires when *every* candidate is invalid.)

**Not factors:** `sigma_max` (0.15 both), temperature schedule (geometric 0.5→0.1 both), the `−ν·C`
budget term (softmax-inert within a step). No outright arithmetic bug; the regressions are
intended-but-harmful features defaulted on.

---

## 3. What I FIXED (and how)

All fixes are **config flags / a new opt-in flag — no rewrite of the core MBD update**. OLD behavior is
recoverable by default-equivalent settings while keeping risk / learned-controller as explicit opt-ins.

| # | regression | fix | mechanism | status |
|---|---|---|---|---|
| 1 | enumerate-all (3.5× wall) | **`fidelity_adaptive` flag** (new, default `True`). `False` ⇒ roll **one fixed fine level** per step (no V̂). | `MBDConfig.fidelity_adaptive`; `step()` branches on it; codesign maps it | ✅ added + verified (`counts={2:K}`; adaptive unit tests still pass; backward-compat intact) — **1.65×** faster |
| 2 | MF-prior pull | `mf_prior_weight=0.0` | existing config flag (override) | ✅ applied in A/B runs |
| 3 | validity collapse | `validity_enabled=false` | existing config flag (override) | ✅ applied in A/B runs |
| 4 | objective flip | testing **`regime_posterior_mode=reward`** (OLD soft-MAX, "search on reward") vs `risk_sensitive` ("search on risk") | existing config flag | 🔬 under A/B test |
| 5 | sigma_max misdiagnosis | revert override **0.7 → 0.15** | I had wrongly set 0.7 earlier ("thread-C" said sigma_max too small — **WRONG**, OLD won with 0.15) | ✅ reverted in A/B runs |

### The "search vs risk" design fork (being measured by A/B)
OLD searched on **reward** (no risk at all) and won. The new thesis needs risk. But risk soft-min in the
**search kernel** weakens MBD's sampling advantage (regression #1A). Two designs under test at
[4,3,4], 6 conflicting regimes, all other fixes applied:
- **A — search on reward, certify/select on risk** (`regime_posterior_mode=reward`): MBD searches the
  signal where it shines; risk ρ_H is applied at certification. Closest to the proven OLD; keeps risk at
  the decision layer.
- **B — search on risk** (`regime_posterior_mode=risk_sensitive`, prior/validity off): keeps risk in the
  kernel; tests whether removing prior/validity alone is enough.

---

## 4. What is NOT yet fixed (backlog)

1. **Blocked-ladder (OLD speed).** The `fidelity_adaptive=false` path rolls a *single fine level* (1.65×
   faster). OLD's blocked ladder (coarse-early/fine-late, 2.5× faster) needs the single jitted `lax.scan`
   restructured into per-block scans (each block has a static `num_env_steps`; a traced per-step level
   can't set `num_env_steps` under jit). Deferred — single-fine is the cleaner signal for the A/B
   comparison anyway.
2. **Faithful adaptive fidelity (the real contribution ②).** Replace enumerate-all with a **cheap V̂
   estimate** (multi-fidelity control-variate, or bandit/UCB that rolls the selected level + occasional
   probes), so `ℓ*=argmax[V̂−νC]` + the dual ν deliver the *compute savings the theory promises*. The
   theory is untouched; only V̂-estimation changes. This is required for contribution ② to be defensible
   (currently adaptive fidelity increases compute).
3. **Config/default cleanup.** `crawling_434.yaml` still carries `sigma_max: 0.7` (residual
   misdiagnosis); `MBDConfig` defaults `mf_prior_weight=0.2`, `validity_enabled=True`,
   `regime_posterior_mode="risk_sensitive"` — reconsider these defaults once the A/B winner is known.
4. **Learned-controller bobbing.** Separately, the learned closed-loop controller reward-hacked a
   forward-velocity warmup objective into bobbing-in-place (net travel ≈ 0). Partial fix applied
   (training objective → net displacement, lines ~507/740), but net-displacement warmup over random
   bodies sits near 0 (honest: random bodies can't travel). The sinusoid controller travels; the learned
   path needs more work. Tracked separately.
5. **Launcher cosmetic.** `run_learned_codesign.py` logs `best_fine_return=None` (the real cert value
   flows into `return_rho_H`); harmless.
6. **CEM-bridge slope parity.** The per-mode dispatch (slope via gravity-tilt / mass / init) is wired
   into OUR marginalizer + verified, but the CEM/CMA-ES bridge (`codesign_env.py`) still varies only
   friction per mode — needed for a fair slope×friction headline.

---

## 5. Results so far

### 5.1 OLD `dc7de15` reproduced (the known-good baseline)
[4,3,4] raw voxel, n_grid64. ours = 4-mode μ∈{0.3,0.4,0.5,0.6}; CEM/CMA-ES single-mode μ=0.45.

| method | return | per-mode (μ=0.3…0.6) | mean | worst | wall |
|---|---|---|---|---|---|
| **ours (MBD)** | **11.66** | [8.37, 12.86, 13.09, 12.31] | 11.66 | 8.37 | 2075 s (35 min) |
| CEM | 4.42 | — | — | — | 175 s |
| CMA-ES | 3.17 | — | — | — | 175 s |

→ **ours / CEM = 2.64×**, matching the prior Table 1 (ours R̄≈10.9, CEM≈6.5, wall 2081/194 s).

### 5.2 NEW [4,3,4], 6 conflicting regimes μ∈{0.05,0.2,0.4,0.7,1.0,1.3}, outer=100/inner=32
All certified by risk ρ_H (soft-min, τ=1) over the 6 modes; mean = plain average; worst = min mode.

| method | risk ρ_H | mean | worst (ice μ=0.05) | per-mode [0.05→1.3] | wall |
|---|---|---|---|---|---|
| **CEM-ρ** (optimizes risk) | **7.17** | 7.89 | **5.82** | [5.82, 8.16, 9.91, 8.35, 7.69, 7.40] | 37 min |
| **CEM-mean** (optimizes mean) | 4.77 | **9.54** | 3.01 | [3.01, 6.29, 12.71, 12.73, 11.52, 10.99] | 37 min |
| ours+sin (PRE-fix: enumerate-all, prior 0.2, validity on, σ=0.7, risk) | 2.41 | — | — | — | 128 min |
| **A: ours search=reward** (fixes: prior 0, validity off, single-fine, σ=0.15) | 3.58 | 8.48 | 1.81 | [1.81, 5.84, 12.71, 11.14, 9.98, 9.39] | 78 min |
| **B: ours search=risk** (same fixes) | **6.54** | 7.10 | 5.24 | [5.24, 7.79, 8.58, 7.30, 6.93, 6.78] | 78 min |

**Reads:**
- **The fixes worked — huge recovery:** ours-risk went **2.41 → 6.54** (2.7×) just from `mf_prior=0` +
  `validity off` (+ single-fine for speed). The MF-prior pull and the validity ESS-collapse were the
  dominant search killers; removing them recovers most of the gap.
- **search=risk (B) ≫ search=reward (A) on the risk metric** (ρ_H 6.54 vs 3.58). B's radar is genuinely
  **balanced/high-floor** [5.24…8.58], worst 5.24 — the risk story. A is **spiky** [1.81…12.71] and
  crashes on ice (1.81). ⇒ **For the risk-robustness thesis you must search on risk; "search reward,
  certify risk" yields a bad-risk_H design (A's design re-certed on risk = 3.58).** That fork is refuted.
- **The risk objective itself works** (independent of ours): CEM-ρ trades a little mean (7.89 vs 9.54)
  for a much higher worst-case floor (**5.82 vs 3.01**) → ρ_H 7.17 vs 4.77. The "why risk" headline.
- **BUT ours-risk (6.54) still trails CEM-ρ (7.17) by ~9%**, and CEM-ρ **dominates B on every mode**
  ([5.82,8.16,9.91,8.35,7.69,7.40] ≥ B elementwise). So on the *same* risk objective, MBD (ours) still
  loses to CEM — which contradicts the theory (MBD > CEM). A residual handicap remains.
- **Likely residual cause:** the risk soft-MIN, even with prior/validity off, *flattens* the denoise
  signal that is MBD's sampling advantage (regression #1A). MBD shines on a *sharp* objective (reward);
  the flat risk surrogate erodes its edge over CEM's elite-selection. OLD won because it searched the
  sharp reward at a *narrow* 4-mode set (no hard ice mode), so the reward-optimal design was naturally
  robust; the 6-mode wide set with μ=0.05 ice breaks that coupling.

### 5.3 Clean regression check — fixed-NEW at OLD's exact setup ✅ DECISIVE
fixed-NEW (reward search, single-fine, prior 0, validity off, σ=0.15) at OLD's **4-mode μ=0.3–0.6**:

| | run-cert (= mean) | per-mode [0.3→0.6] | wall |
|---|---|---|---|
| **OLD `dc7de15`** | 11.66 | [8.37, 12.86, 13.09, 12.31] | 35 min |
| **fixed-NEW (reward, 4-mode)** | **11.36** | [8.58, 11.50, 12.98, 12.38] | 51 min |

**fixed-NEW (11.36) ≈ OLD (11.66), within ~3%, per-mode nearly coincident.** ⇒ **the regression is
100% explained and fixed**: with equivalent config, NEW reproduces OLD. (Wall still 51 vs 35 min because
single-fine ≠ OLD's blocked ladder — a known, deferred speed item, not a correctness gap.)

**Verdict (final):**
- **Fixes complete.** The 4 regressions (MF-prior 0.2, validity collapse, risk-soft-min search
  objective, enumerate-all wall) fully account for the NEW-vs-OLD gap. MBD's search advantage is intact
  and **MBD ≫ CEM on the (sharp) reward objective** (11.36 vs OLD-CEM 4.42 ≈ 2.6×).
- **The 6-mode `ours-risk 6.54 < CEM-ρ 7.17` is NOT a code regression.** It is (a) the 6-mode wide-+ice
  set is genuinely harder, and (b) the **risk soft-MIN flattens the search signal**, eroding exactly the
  sampling edge that lets MBD beat CEM on a sharp objective. CEM's elite-selection is less sensitive to
  signal sharpness, so on a flat risk surrogate CEM catches up.
- **Path to "ours > CEM" under risk (design, not bug):** give the MBD search a *sharp enough* but
  *worst-case-aware* signal. Candidates to test: (i) milder risk temperature τ (between mean and worst —
  τ=1 full soft-min is the flattest), (ii) search on reward/mean but with a robustness regularizer,
  (iii) the 4-mode-style narrow-conflict regimes where the reward-optimal design is naturally robust.
  The decision-layer risk certification (ρ_H) is kept regardless.

---

## 5.4 ⚠️ CRITICAL: the OLD "ours ≫ CEM" was vs a WEAK CEM code path

Re-certifying the **NEW** CEM/CMA-ES (run via `run_baseline_p2` = `codesign_env` bridge + `cem_jax`
vmap) at the **same 4-mode [0.3–0.6] mean** metric as ours:

| method (4-mode [0.3–0.6], mean) | mean | per-mode | worst |
|---|---|---|---|
| ours-NEW (reward, MBD) | 11.36 | [8.58, 11.50, 12.98, 12.38] | 8.58 |
| **NEW CEM** (bridge+cem_jax) | **11.33** | [9.05, 12.05, 12.61, 11.61] | 9.05 |
| **NEW CMA-ES** | **12.92** | [9.23, 13.98, 14.82, 13.64] | 9.23 |

vs OLD CEM (`run_co_design`, standard framework): single-mode cert **4.42**, table 4-mode **6.52**.

**The NEW (correctly-vmapped) CEM is ~2× stronger than the OLD-framework CEM.** ⇒ **the prior "ours ≫
CEM (2.64×)" was an artifact of comparing strong-MBD against a weak/slow OLD-CEM code path.** On a fair
comparison with the strong bridge+cem_jax baselines, **ours ≈ CEM and CMA-ES > ours** at 4-mode
locomotion. Combined with §5.2 (6-mode: ours-risk 6.54 < CEM-ρ 7.17), **ours does not currently beat
fair baselines on any tested setting.** The MBD-vs-OLD-CEM regression was real and is fixed; but the
*headline competitive claim itself* needs re-examination against the strong baselines.

Caveat: NEW CEM here optimized single-mode μ=0.45 and generalized to the narrow [0.3–0.6] band (lucky on
a narrow set); a wider/conflicting regime set is where multi-mode ours *should* separate — but the
6-mode result shows CEM-ρ (multi-mode CEM) still beats ours-risk there.

### 5.4.3 RESOLVED (2×2) — init_std=0.3 IS the cause; OLD code path is fragile to it
Ran the OLD repo (`dc7de15`, PYTHONPATH override) CEM env200 (200 steps, single-mode μ=0.45) at both
init_std, re-certified at 4-mode [0.3–0.6] mean:

| CEM, 4-mode mean | init_std=0.3 | init_std=0.8 | Δ(init_std) |
|---|---|---|---|
| **OLD code path** (cem_baseline.py) | **6.49** (= the prior table's CEM\|m\|=200 row [4.21,7.50,7.45,6.81], reproduced exactly) | **10.36** | **+60%** |
| **NEW code path** (cem_jax) | 9.81 | 11.33 | +16% |

**Resolution:** init_std=0.3 IS the dominant cause. The **OLD CEM code path is highly fragile to a narrow
init_std** (6.49→10.36, +60% just from 0.3→0.8); the NEW path is *robust* to it (9.81 at 0.3) — and that
robustness is what made §5.4.2's "NEW CEM at 0.3 = 9.81 ⇒ init_std minor" misleading (it masked the
effect that dominates in the OLD path). At a properly-tuned init_std=0.8, **OLD CEM (10.36) ≈ NEW CEM
(11.33) ≈ ours (11.36)**. ⇒ **The prior "ours ≫ CEM/CMA-ES" was an under-tuned-baseline artifact
(init_std/σ0=0.3).** With matched proper exploration, ours ≈ CEM and CMA-ES (12.92) > ours.

### 5.4.2 [intermediate, superseded by 5.4.3] — first thought code path, not init_std
A controlled run disproved the init_std hypothesis below. NEW CEM (bridge+cem_jax), single-mode μ=0.45,
200 steps: **init_std=0.3 → 10.69**, init_std=0.8 → 13.08 (only ~1.2×). So in the NEW code path init_std
is a *minor* knob. Yet OLD CEM (`run_co_design` standard framework, init_std 0.3, 200 steps) = 4.42–6.49.
⇒ **The OLD CEM weakness (~2×) is the OLD CEM code path itself (the standard-framework / legacy
evaluate_batch CEM), not the exploration std.** The fair baseline is the NEW bridge+cem_jax CEM, which is
strong (10.69–13.08) and robust to init_std. The prior "ours ≫ CEM" was vs the weak OLD-codepath CEM.
(§5.4.1 below is superseded on the *cause*; its numbers stand.)

### 5.4.1 [SUPERSEDED cause] OLD baselines weak — first hypothesis: narrow exploration std
The OLD baselines (`configs/soft_robot/baselines/{cem,cmaes}_crawling.yaml`) used the **same 80-d phi**
as OLD ours (theta_len=128 = 48 voxel + 80 phi — *not* a controller-dim mismatch) and the **same 200
env-steps** (the user confirmed OLD CEM was run at 200 steps; the |m|=200 table row). The single cause:
- **Narrow exploration std:** CEM `init_std=0.3`, CMA-ES `sigma0=0.3` — vs a proper **0.8**. Too-small
  exploration ⇒ the ES contracts into a poor local optimum. At the *same* 200 steps: OLD CEM(|m|=200)
  4-mode ≈ **6.49** (table) vs **NEW CEM(init_std 0.8) 11.33** (≈1.7×); single-mode OLD 4.42 vs NEW 13.08
  (≈3×). CMA-ES identical (σ0=0.3 → 3.17; 0.8 → 14.86).

⇒ **The prior "ours ≫ CEM/CMA-ES (2.64×)" is an artifact of under-explored (`init_std/σ0=0.3`)
baselines — not MBD optimizer superiority.** With a properly-tuned exploration std (0.8), CEM ≈ ours and
CMA-ES > ours (§5.4). (Single-knob confirmation — NEW CEM at init_std=0.3 vs 0.8, same 200 steps — running.)

### Bottom line for the campaign
- The NEW-vs-OLD **code regression is fixed** (§5.3) — that part is solid.
- But the **competitive claim "ours > CEM/CMA-ES" does not survive fair baselines.** It must be earned,
  not inherited from the OLD (handicapped-baseline) numbers. Candidate edges still to test: (a) the
  risk/worst-case angle at a tuned τ (sweep running), (b) a deceptive/multimodal landscape where MBD's
  global sampling beats ES local search, (c) multi-fidelity *compute* savings (needs the cheap-V̂ impl).

## 5.5 ⚠️⚠️ FINAL VERDICT — MBD and ES have OPPOSITE exploration optima; ours does not beat tuned baselines

Raising ours' exploration (sigma_max 0.15→0.7, to "match" CEM's init_std 0.8) **hurt** ours:
- ours-reward 4-mode: 11.36 (σ_max 0.15) → **8.86** (σ_max 0.7)
- ours-risk 6-mode: 6.54 (0.15) → **5.59** (0.7)

Because MBD's denoise is a **reward-weighted mean of proposals around the current center** — it needs
*local* (narrow) proposals; too-wide proposals make the weighted mean diffuse and convergence worse. ES
(CEM/CMA-ES) is the opposite: it needs *wide* exploration for its elite selection. **So "matched
exploration std" is the wrong fairness criterion — each optimizer must use its own optimum.**

### Each optimizer at its own tuned-best (4-mode [0.3–0.6] mean)
| optimizer | best exploration | best value |
|---|---|---|
| **ours (MBD)** | narrow (σ_max 0.15) | **11.36** |
| **CEM** | wide (init_std 0.8) | 11.33 |
| **CMA-ES** | wide (σ0 0.8) | **12.92** |

**Verdict: with each baseline individually tuned, ours (11.36) ≈ CEM (11.33) and CMA-ES (12.92) > ours.**
6-mode risk likewise: ours-risk best 6.54 < CEM-ρ 7.17. **ours does not beat fairly-tuned baselines on the
[4,3,4] crawling task.** The smooth/near-convex crawling landscape saturates ~11–13 for every optimizer at
its optimum, so MBD's global-sampling advantage cannot show. The prior "ours ≫ CEM/CMA-ES" was entirely an
artifact of under-tuned baselines (init_std/σ0=0.3), to which the OLD CEM code path was especially fragile.

### Where ours could still legitimately win (next experiments)
1. **Deceptive / multimodal landscape** — a task/terrain where ES contracts into a local optimum and MBD's
   diffusion sampling escapes it. This is MBD's theoretical edge and the [4,3,4] crawl doesn't exercise it.
2. **Sample efficiency at fixed small budget** — MBD reaches the optimum with fewer evals / narrower
   exploration; quantify the budget-vs-quality curve (MBD may dominate at low budget even if both saturate).
3. **Multi-fidelity compute savings** — needs the faithful cheap-V̂ adaptive impl (§4.2), then compare
   wall/return frontier vs full-fidelity ES.
4. **Risk-robustness on conflicting regimes** — only meaningful once the search is competitive; currently
   ours-risk < CEM-ρ.

## 6. Open questions / next steps

1. **Wait on B** → fill the table → decide the search objective.
2. **Clean regression check (recommended):** run the **fixed NEW** at OLD's exact 4-mode μ=0.3–0.6 and
   confirm it reproduces ≈11.66. This isolates "regression fully fixed?" from "6-mode is just harder".
3. If ours still trails at matched objective → diff the *survivor selection* + *fine-revalidation* paths
   (OLD top-K by mean vs NEW by ρ*; cert decode parity) for a remaining gap.
4. Implement backlog #1 (blocked ladder) + #2 (cheap-V̂ adaptive) for the campaign rerun + contribution ②.

---

## Appendix — reproduce

```bash
# OLD (known-good), isolated worktree, OLD imports:
cd /workspace/gd_old && PYTHONPATH=/workspace/gd_old CUDA_VISIBLE_DEVICES=0 \
  python scripts/tasks/soft_robot/co_design/main/run_co_design.py \
  configs/soft_robot/main/crawling_ground.yaml --seed 0      # ours  -> results/.../main/crawling_ground/
  # …/baselines/cem_crawling.yaml ; …/baselines/cmaes_crawling.yaml

# NEW fixed A/B (6-mode, fixes on):
cd /workspace/genedynamics && bash /tmp/run_AB_variants.sh    # regime_posterior_mode=reward | risk_sensitive
#   common: controller_type=sinusoid K=100 M=32 num_modes=6 mf_prior_weight=0.0 \
#           validity_enabled=false fidelity_adaptive=false sigma_max=0.15
# raw-voxel risk re-cert: python /tmp/recert_434.py <design.json>
```
