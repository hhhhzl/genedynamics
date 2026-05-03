# Soft Robot Co-Design — Phases 1-5 Runbook

This folder is the entry point for the writeup
*Regime-Robust Anytime-Fidelity Model-Based Diffusion for Soft Robot Co-Design
with Modern 3D Morphology Priors*. Phases 1-4 ship the implementation; Phase 5
runs the experimental sweeps that produce the paper's tables and figures.

---

## 1. Phase status

| Phase | Scope | Status | Where it lives |
|---|---|---|---|
| **Phase 0** | Robotization extraction (`SoftBodySpec` + `default_robotize`); third_party stubs | ✅ done | [genedynamics/morphology/](../../genedynamics/morphology/), [third_party/](../../third_party/) |
| **Phase 1** | `composite + alm_adaptive` scheduler hookup; JM2D u-step inner denoising; alm history in metadata | ✅ done | [main_v2/crawling_alm.yaml](main_v2/crawling_alm.yaml), [solvers/single/mrmfmbd/](../../genedynamics/solvers/single/mrmfmbd/) |
| **Phase 2** | Terrain height-fields; kinematic manipuland (push); regime bank; risk-sensitive marginalization | ✅ done | [main_v2/locomotion_terrain.yaml](main_v2/locomotion_terrain.yaml), [main_v2/push.yaml](main_v2/push.yaml), [envs/external/jax_mpm/{terrain,manipuland,tasks}/](../../genedynamics/envs/external/jax_mpm/) |
| **Phase 3** | Mesh-pipeline robotization; asset bank manifest; CLI (`build_asset_bank` / `robotize_bank`); `random_shapes` + `triposg` (lazy) priors | ✅ done | [main_v2/crawling_from_mesh.yaml](main_v2/crawling_from_mesh.yaml), [genedynamics/morphology/](../../genedynamics/morphology/), [scripts/morphology/](../../scripts/morphology/) |
| **Phase 4** | SHAC-only baseline (truncated h + critic + Polyak); MBD+SHAC top-K refinement (writeup §8.1); finite-grad guard | ✅ done | [main_v2/shac_only_crawling.yaml](main_v2/shac_only_crawling.yaml), [main_v2/crawling_alm_shac.yaml](main_v2/crawling_alm_shac.yaml), [solvers/single/shac/](../../genedynamics/solvers/single/shac/) |
| **Phase 5** | Full ablation sweeps (writeup Tables 1-4) + figures + writeup wrap | ⏳ to run | this README §3-§5 |

All v2 configs round-trip through `BaselineConfig`; the legacy `main/crawling_ground.yaml` still works unchanged.

> ⚠ **Two prerequisite gaps** must be closed before Phase 5 sweeps produce the writeup's main artifacts. See **[§1.5 Setup gaps](#15-setup-gaps-must-close-before-phase-5)** below for the exact commands.

---

## 1.5 Setup gaps (must close before Phase 5)

Phases 1-4 ship the implementation, but two pieces of *infrastructure* are documented-but-not-installed. They're separated from the code phases because they each require a one-time external download / new code that wasn't in scope for the algorithm work.

### Gap A — TripoSG is documented, NOT cloned

[third_party/morphology_priors/triposg/](../../third_party/morphology_priors/triposg/) currently contains only a README. The adapter at [genedynamics/morphology/priors/triposg.py](../../genedynamics/morphology/priors/triposg.py) is fully written (lazy import + 3 fallback `from_pretrained` paths), but `get_prior("triposg")` raises `MissingDependencyError` until the upstream package is on the Python path.

**Why we did NOT auto-clone in Phase 0.2**:
- Repo + model weights ≈ 5-8 GB → shouldn't silently consume disk
- HuggingFace weights may need a token
- CI / dev loop works on `random_shapes` prior alone
- Upstream API churn — defer the version pin to actual install time

**One-shot install (do this before §3.5 Table 3 prior comparison)**:
```bash
cd third_party/morphology_priors/triposg
git clone https://github.com/VAST-AI-Research/TripoSG.git .
pip install -r requirements.txt
# Then verify:
python -c "from genedynamics.morphology.priors import get_prior; \
           p = get_prior('triposg'); print(p.metadata)"
```

What this unlocks: the writeup's main 3D prior. Without it, Table 3 ships with 1 prior (`random_shapes` as the legacy placeholder) instead of 4.

**TRELLIS / Hunyuan3D-2 / Point-E** (other Table 3 rows): all need new adapter files at `genedynamics/morphology/priors/<name>.py`. The TripoSG adapter is the template — each new prior is ~150 lines of `from_pretrained` + `_coerce_to_trimesh`.

### Gap B — DiffuseBot-quality renderer NOT yet built

What we have today: [run_co_design_with_gif.py](../../scripts/tasks/soft_robot/co_design/main/run_co_design_with_gif.py) renders an animated GIF + a 2-frame summary PNG using matplotlib voxel-cube glyphs. Output ≈ 3/10 paper quality — fine for sanity-checking gait, not for the writeup's Figure 4.

What the writeup needs (DiffuseBot Figure 4 style):
- Particle-cloud render (every MPM particle, not just 24 voxel cubes)
- Multi-checkpoint horizontal grid: 5 columns showing the body at epochs 1, K/4, K/2, 3K/4, K
- Per-actuator color (red / green / blue / yellow groups)
- Wood-floor texture + perspective camera + soft directional lighting
- Optional contact shadow

| Recommended path | Install | Wall-time | Quality |
|---|---|---|---|
| **PyVista** (start here) | `pip install pyvista` (~50 MB) | ½ day | 6.5/10, paper-draft adequate |
| Polyscope | `pip install polyscope` | ½ day | 7/10 |
| pyrender + trimesh (pyrender already installed) | 0 | 1 day | 7/10, manual scene compose |
| Mitsuba 3 | `pip install mitsuba` (~200 MB) | 2 days | 9/10, path-traced |
| Blender (subprocess) | install Blender + py script | 3-4 days | 10/10, what DiffuseBot likely used |

**Recommended first cut** — `scripts/visualizations/render_soft_robot_checkpoints.py`:
1. Inputs: `results/<exp>/results_seed_*.json` (Phase 1 already saves `theta_history` of shape `(K, D)`)
2. For each chosen checkpoint, re-rollout `(x_k, phi_k)` capturing per-particle positions
3. PyVista off-screen render: particles as point cloud colored by `actuator_id`, programmatic wood plane, key + fill light
4. Composite 5-column PNG → `results/<exp>/figures/evolution.png`

Then `make_figure_evolution.py` calls this for each Table-1 method and produces the writeup Figure 4 grid.

**Decision required** before writing this:
- Stop at PyVista (paper-draft) or commit to Blender (camera-ready)?
- Want a per-asset `render_asset_bank.py` too (lets us include §3.5 prior comparisons in Figure 2 qualitative grid)?

---

## 2. Config registry

| Config | Method | Phase | Writeup row | Notes |
|---|---|---|---|---|
| [main/crawling_ground.yaml](main/crawling_ground.yaml) | MRMFMBD | 0 | Table 1 "MBD" baseline | Legacy entry; nothing changed |
| [main_v2/crawling_alm.yaml](main_v2/crawling_alm.yaml) | MRMFMBD + composite scheduler | 1 | Table 1 "MBD + adaptive iALM" | `nu_max=0.0` → tracker inert; flip to ablate |
| [main_v2/locomotion_terrain.yaml](main_v2/locomotion_terrain.yaml) | MRMFMBD on 12 terrain regimes | 2 | Table 1 + Table 4 | `regime_posterior_mode: risk_sensitive` |
| [main_v2/push.yaml](main_v2/push.yaml) | MRMFMBD on 36 push regimes | 2 | Table 1 push column | Kinematic AABB box |
| [main_v2/crawling_from_mesh.yaml](main_v2/crawling_from_mesh.yaml) | MRMFMBD on a mesh-derived body | 3 | Table 3 prior comparison | Set `softbody_spec_path:` after building a bank |
| [main_v2/shac_only_crawling.yaml](main_v2/shac_only_crawling.yaml) | SHAC | 4.1 | Table 1 #9 / Table 2 / Q3 | Controller-only on default body |
| [main_v2/crawling_alm_shac.yaml](main_v2/crawling_alm_shac.yaml) | MBD + SHAC refinement | 4.2 | Table 1 #11 | `shac_refine_steps: 5` (writeup §8.1) |
| [ablation/no_fidelity_ladder_env*.yaml](ablation/) | MRMFMBD with ladder off | 0 | Table 4 fixed-level | env100 / env200 fix the eval at one fidelity |
| [ablation/no_mode_marginalization.yaml](ablation/no_mode_marginalization.yaml) | MRMFMBD num_modes=1 | 0 | Table 1 ablation | Reward-only marginalization |
| [baselines/{cem,cmaes}_crawling*.yaml](baselines/) | CEM / CMA-ES | 0 | Table 1 #2-#3 | Same evaluator, gradient-free |

**Knobs you toggle WITHOUT touching code** (all under `method_params:` unless noted):

| Knob | Where | Phase | Effect |
|---|---|---|---|
| `inner_denoise_steps` | method_params | 1.3 | JM2D u-step refinement; 0 → off, 5-10 → ablation |
| `regime_posterior_mode` | method_params | 2.3 | `"reward"` (legacy) / `"risk_sensitive"` (writeup §5) |
| `risk_temperature` | method_params | 2.3 | `tau_r → 0` = max-min, `tau_r → ∞` = uniform |
| `regime_bank_kind` | evaluator_runtime | 2.2 | `"train_locomotion" / "test_locomotion" / "train_push" / "test_push"` |
| `softbody_spec_path` | evaluator_runtime | 3 | Load mesh-derived `SoftBodySpec.npz` instead of `default_robotize` |
| `shac_refine_steps` | method_params | 4.2 | 0 → no SHAC refinement; 5 → writeup §8.1 hybrid |
| `nu_max` | scheduler_config.constraint_schedulers[0] | 1.2 | 0 → ALM tracker inert; >0 → activate budget penalty |

---

## 3. Phase 5 — step-by-step playbook

Phase 5 is execution: producing the writeup's Tables 1-4 and Figures 1-6. The
code is all in place; the work is sweeps, aggregation, and plots.

### 3.0 — Pre-flight (do this once)

```bash
# 1. Sanity: configs parse and the baseline registry sees everything.
python -c "
import yaml
from genedynamics.experiments.framework.baselines import (
    MRMFMBDBaseline, CMAESBaseline, CEMBaseline, SHACBaseline,
)
for fn in [
    'configs/soft_robot/main/crawling_ground.yaml',
    'configs/soft_robot/main_v2/crawling_alm.yaml',
    'configs/soft_robot/main_v2/locomotion_terrain.yaml',
    'configs/soft_robot/main_v2/push.yaml',
    'configs/soft_robot/main_v2/shac_only_crawling.yaml',
    'configs/soft_robot/main_v2/crawling_alm_shac.yaml',
]:
    yaml.safe_load(open(fn))
print('all configs parse OK')
"

# 2. Single-seed smoke run on the legacy baseline (~5-10 min on a 24GB GPU).
python scripts/tasks/soft_robot/co_design/main/run_co_design_with_gif.py \
    configs/soft_robot/main/crawling_ground.yaml
ls results/soft_robot/main/crawling_ground/   # expect: results.json + .gif

# 3. Phase 4 SHAC smoke (~15 min on a 24GB GPU; CPU-OK with smaller h/N).
python scripts/tasks/soft_robot/co_design/main/run_co_design_with_gif.py \
    configs/soft_robot/main_v2/shac_only_crawling.yaml
```

If any of the above fail, **stop and debug** before running sweeps — the same
failure will kill 100+ hours of compute.

### 3.1 — Phase 3 prerequisite: build the asset bank (writeup Table 3)

Skip this section if you're only reproducing Tables 1, 2, 4 (default body).

```bash
# (a) Generate meshes from a prior. random_shapes for dev, triposg for the paper.
python scripts/morphology/build_asset_bank.py \
    --prior random_shapes \
    --bank-name loco_v1 \
    --prompts-file scripts/morphology/prompts/locomotion.txt \
    --n-per-prompt 30 --seed 0
# → data/asset_banks/loco_v1/{manifest.json, meshes/}

# (b) Robotize the bank.
python scripts/morphology/robotize_bank.py --bank-root data/asset_banks/loco_v1
# Last stdout line: "RobotizationSuccess (whole bank): XX.X%"
# Writeup target: ≥30 %. We see ~100% with random_shapes, ~50-70% expected with triposg.

# (c) Pick the top-N by particle count and write per-asset configs.
python -c "
import json, os, shutil
m = json.load(open('data/asset_banks/loco_v1/manifest.json'))
ok = sorted([a for a in m['assets'] if a['robotized']],
            key=lambda a: -a['report']['n_filled_cells'])[:5]
for a in ok:
    cfg = open('configs/soft_robot/main_v2/crawling_from_mesh.yaml').read()
    cfg = cfg.replace(
        '# softbody_spec_path: data/asset_banks/loco_v1/robotized/<asset_id>.npz',
        f'softbody_spec_path: data/asset_banks/loco_v1/robotized/{a[\"id\"]}.npz',
    )
    out = f'configs/soft_robot/main_v2/from_mesh_{a[\"id\"]}.yaml'
    open(out, 'w').write(cfg)
    print(out)
"
```

For TripoSG (writeup main prior): clone `third_party/morphology_priors/triposg`
per its README, then swap `--prior random_shapes` → `--prior triposg` in (a).
The remaining steps are unchanged.

### 3.2 — Sweep runner

The repo doesn't yet have a sweep CLI; here's a minimal one to drop in
`scripts/sweeps/run_sweep.py`:

```python
#!/usr/bin/env python
"""Run one config across a set of seeds; one process per (config, seed)."""
import argparse, subprocess, sys, time, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("configs", nargs="+")
ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

seeds = [int(s) for s in args.seeds.split(",")]
runner = "scripts/tasks/soft_robot/co_design/main/run_co_design_with_gif.py"
for cfg in args.configs:
    for seed in seeds:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "GENEDYNAMICS_SEED_OVERRIDE": str(seed)}
        cmd = [sys.executable, runner, cfg]
        print(f"[{time.strftime('%H:%M:%S')}] seed={seed} cfg={cfg}")
        if not args.dry_run:
            subprocess.run(cmd, env=env, check=True)
```

Note: the existing `run_co_design.py` reads `seeds` from the YAML; the env
override needs to be respected by the runner. If it isn't, use a per-seed
config file (the simplest fix for the writeup pipeline).

### 3.3 — Table 1: Main end-to-end co-design (writeup §16)

**11 methods × 3 tasks × 5 seeds = 165 runs**. Estimated 660-1300 GPU-hours
on a single A6000 / A100 80GB.

| # | Method | Config | Notes |
|---|---|---|---|
| 1 | Random prior | needs new config | `cem_crawling.yaml` with 1 generation, popsize=1 |
| 2 | CEM | [baselines/cem_crawling*.yaml](baselines/) | Already exists |
| 3 | CMA-ES | [baselines/cmaes_crawling*.yaml](baselines/) | Already exists |
| 4 | MBD (vanilla) | [main/crawling_ground.yaml](main/crawling_ground.yaml) | Legacy |
| 5 | MBD + regime | locomotion_terrain.yaml with `regime_posterior_mode: risk_sensitive` | ✓ |
| 6 | MBD + adaptive iALM | crawling_alm.yaml with `nu_max: 5.0` | Set in YAML |
| 7 | DiffuseBot-style | needs new config | DiffuseBot-style: PointE prior + gradient co-design |
| 8 | DiffAqua-style | needs new config | low-D shape interp + neural controller |
| 9 | SHAC-only | [main_v2/shac_only_crawling.yaml](main_v2/shac_only_crawling.yaml) | ✓ |
| 10 | **Ours** (full) | locomotion_terrain.yaml + nu_max>0 + risk_sensitive | ✓ |
| 11 | Ours + SHAC refine | [main_v2/crawling_alm_shac.yaml](main_v2/crawling_alm_shac.yaml) | ✓ |

For each row × {locomotion, push, carry}, run all 5 seeds. Locomotion uses
`task_id: locomotion`; push uses `task_id: push`; carry needs Phase 5.4 work
(see §3.6 below).

```bash
# Example: run Method 10 (Ours) on locomotion across 5 seeds.
python scripts/sweeps/run_sweep.py \
    configs/soft_robot/main_v2/locomotion_terrain.yaml \
    --seeds 0,1,2,3,4
```

Each successful run writes `results/soft_robot/main_v2/<exp>/results_seed_*.json`.

### 3.4 — Table 2: Controller-only (writeup §13.2 / Q3)

Fix morphology, sweep controller optimizers. **8 methods × 3 tasks × 5 seeds = 120 runs**.

The fixed morphology is the optimal `x` from a successful Table 1 run. Save
it, then point the new config at it via `morphology: [...]` in `method_params`.

```bash
# Extract best morphology from a Table 1 result.
python -c "
import json, numpy as np
r = json.load(open('results/soft_robot/main_v2/locomotion_terrain/results_seed_0.json'))
x = np.asarray(r['x']).tolist()
print('morphology:', x)
" > best_morph.txt

# Use it in shac_only_crawling.yaml by adding under method_params:
#   morphology: [<paste here>]
```

Suggested config naming for clarity:
`controller_only_<method>.yaml`. Methods: cem, cmaes, mbd, diffusebot,
diffaqua, shac, ours, ours+shac.

### 3.5 — Table 3: Modern prior + robotization (writeup §16)

**4 priors × 2 tasks × 5 seeds = 40 runs** + asset bank generation cost.

> ⚠ Blocked by **[§1.5 Gap A](#gap-a--triposg-is-documented-not-cloned)**. TripoSG is the writeup's main prior — the adapter is written but the upstream package is not cloned. TRELLIS / Hunyuan3D-2 / Point-E need new adapter files (use [triposg.py](../../genedynamics/morphology/priors/triposg.py) as the template, ~150 lines each).

| Prior | Adapter status | Clone status |
|---|---|---|
| Point-E (legacy) | ❌ no adapter | ❌ no clone (or substitute `random_shapes`) |
| **TripoSG** (main) | ✅ [adapter ready](../../genedynamics/morphology/priors/triposg.py) | ❌ **clone needed** — see §1.5 Gap A |
| TRELLIS | ❌ no adapter | ❌ no clone |
| Hunyuan3D-2 | ❌ no adapter | ❌ no clone |

For Phase 5 minimum, you can ship Table 3 with just:
- `random_shapes` as the "legacy" placeholder
- TripoSG (after closing Gap A)

Run §3.1 to build a per-prior asset bank, then run `crawling_from_mesh.yaml`
on each top-K successful asset and aggregate.

### 3.6 — Table 4: Anytime iALM ablation (writeup §16 Table 4)

**7 fidelity schemes × 1 task × 5 seeds = 35 runs**. The schemes:

| Row | Knob change in [crawling_alm.yaml](main_v2/crawling_alm.yaml) |
|---|---|
| Fixed-low (ℓ=30) | `num_fidelity_levels: 1`, `fine_fidelity_level: 0` |
| Fixed-mid (ℓ=100) | `num_fidelity_levels: 1`, `fine_fidelity_level: 1` |
| Fixed-high (ℓ=300) | `num_fidelity_levels: 1`, `fine_fidelity_level: 2` |
| Coarse-to-fine | `num_fidelity_levels: 3`, `fidelity_ladder_type: geometric` |
| Adaptive-global | + `nu_max: 5.0`, leave I_max/topK_max as scheduler picks |
| Adaptive-candidate | (TODO Phase 5.4 — needs per-candidate fidelity, not implemented yet) |
| Adaptive-regime-candidate | (TODO Phase 5.4 — same) |

The first 5 rows are pure config edits. The last 2 require a small extension
to the MBD scan body (per-candidate `ℓ_{j,m}` → per-candidate cost) — left
explicitly for Phase 5 because it's the most invasive change in the table.

### 3.7 — Carry task (writeup §11 Task 3, optional)

Currently `manipuland.py` has `horizontal_only: True` (push regime). Carry
needs:
- Vertical box motion enabled (flip the flag)
- Contact stability score `G_T` reward implementation
- A new `tasks/carry.py` reward wrapper

Estimated 1-2 days of work; the writeup permits leaving carry to the appendix.

### 3.8 — Aggregation + figures

> ⚠ Figure 4 (qualitative evolution grid, DiffuseBot-style) is blocked by **[§1.5 Gap B](#gap-b--diffusebot-quality-renderer-not-yet-built)**. Current matplotlib voxel renderer (the existing `crawling_best.gif` you see in `results/.../crawling_ground/`) is paper-draft quality only.

Existing analysis scripts under [scripts/tasks/soft_robot/co_design/analysis/](../../scripts/tasks/soft_robot/co_design/):

| Script | What it does | Maps to writeup |
|---|---|---|
| `analyze_main_result.py` | Mean / std across seeds for one config | Table 1 cell values |
| `compare_4_methods.py` | Bar plot of 4 methods on one task | Figure 6 (controller bar) |
| `compare_fidelity_ablation.py` | Fidelity scheme comparison plot | Figure 5 (selected ℓ levels) |
| `analyze_diffusion_run.py` | Per-step diagnostics (T_k, ESS, w_c) | Figure 5 overlay |
| `compare_reward_curves.py` | Reward-vs-step curves | Figure 4 (reward curves only) |
| `analyze_baseline_run.py` | Single-method deep dive | sanity |

Missing scripts (need to write):

| Script | Status | Maps to writeup |
|---|---|---|
| `scripts/visualizations/render_soft_robot_checkpoints.py` | ❌ not written — see §1.5 Gap B for design | Figure 4 (qualitative evolution grid) |
| `scripts/visualizations/render_asset_bank.py` | ❌ not written — uses same renderer as Gap B | Figure 2 row 1 (generated 3D asset) |
| `make_table1.py` | ❌ not written | Table 1 (wide CSV across all 11 methods × 3 tasks) |
| `make_table4.py` | ❌ not written | Table 4 (fidelity ablation table) |
| `make_figure3.py` | ❌ not written | Figure 3 (terrain regime heatmap) |
| `make_figure_pipeline.py` | ❌ not written | Figure 1 (dataflow diagram, can be hand-drawn) |

---

## 4. Wall-clock estimates

A6000 (48GB) baseline:

| Workload | Per-seed wall | × 5 seeds | × all-rows |
|---|---|---|---|
| Table 1 row 4 (vanilla MBD, locomotion) | ~30 min | 2.5 h | — |
| Table 1 row 10 (Ours, locomotion) | ~45 min | ~4 h | ~12 h (× 3 tasks) |
| Table 1 row 9 (SHAC-only, locomotion) | ~3 h | ~15 h | ~45 h (× 3 tasks) |
| Table 1 row 11 (Ours+SHAC, locomotion) | ~50 min | ~4.5 h | ~13 h |
| Table 1 full (locomotion only) | — | — | ~50-80 h |
| Table 1 full (3 tasks × 11 rows × 5 seeds) | — | — | **~250-400 h** |
| Asset bank generation (50 meshes, TripoSG) | ~1 h offline | once | ~1 h |
| Table 3 (4 priors × 2 tasks × 5 seeds) | ~30 min | — | ~25 h |
| Table 4 (7 schemes × 5 seeds) | ~30-60 min | — | ~25-50 h |

Total Phase 5 compute: **~300-500 GPU-hours** for a complete writeup.
On 4 × A100 80GB this is ~3-5 days wall.

---

## 5. Debugging guide

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'shac' not registered` | Stale `__pycache__` | `find . -name __pycache__ -exec rm -rf {} +` |
| MPM rollout produces NaN reward | Friction kernel `0/0` (pre-Phase 4) | Already patched in `grid_op_3d`; if still happens, lower `cfg.dt` |
| SHAC `actor_grad_norm = nan` | h too long or actor_lr too high | `h: 16`, `actor_lr: 1e-3`, `actor_grad_clip: 0.5` |
| MBD+SHAC hurts vs MBD alone | Refinement overshoots | `shac_refine_lr: 1e-4`, `shac_proximal_lambda: 10.0` |
| Push reward stuck at -50 | Object never moves; sticky penalty fires | Lower `mass_o` in regime bank or move `push_goal_x` closer |
| RobotizationSuccess < 30% | Mesh is too thin / disconnected | Tune `MeshRobotizeConfig.min_filled_cells` down, or repair fewer meshes |
| OOM at 32 candidates × 36 push regimes | Too many parallel rollouts | Drop `M_k: 16` or sub-sample regime bank in adapter |
| Per-step JIT recompile every episode | `h` or `n_envs` changing across calls | They're closed-over Python ints; should be impossible. Check no upstream code is mutating `SHACConfig` |

---

## Appendix A: Controller architecture

The phi-as-controller spec is unchanged across Phases 1-4. Pasted from the
original README for completeness:

```
raw_i(t) = tanh( W_i · sin(ωt + phases) + b_i + g_i · (1000 · v_com_x) )
act_i(t) = raw_i(t) · σ( a_i + c_i · t/T )
```

| Component | Shape | Role |
|---|---|---|
| `W` | (10, 4) | Per-actuator sin-wave weight matrix |
| `b` | (10,) | Bias |
| `g` | (10,) | Per-actuator velocity feedback gain |
| `a` | (10,) | Sigmoid-envelope offset |
| `c` | (10,) | Sigmoid-envelope slope |

`phi = [W.flatten(), b, g, a, c]` → 80 parameters. `phi ∈ [-0.5, 0.5]`.
`ω = 20.0 rad/s`, `frame_dt = 8 ms` → period ≈ 39 env steps.

---

## Appendix B: Morphology design

- Voxel grid: 4×3×4 = 48 voxels (configurable via `evaluator_runtime.voxel_dims`)
- Z-axis symmetry: optimizer sees 4×3×2 = 24 dims, mirrored before rollout
- Occupancy range: `x_lo=0.2` to `x_hi=1.0` (5:1 density ratio)
- Prior: `x_mean=0.6`, `x_std=0.3`

For Phase 3 mesh-derived bodies, voxel grid drops to 3×3×3 = 27 (matches
`MeshRobotizeConfig` default) and z-symmetry is disabled (mesh shapes are
not symmetric in general). See `crawling_from_mesh.yaml` comments.
