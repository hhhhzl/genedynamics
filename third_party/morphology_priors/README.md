# 3D Foundation Morphology Priors

Modern 3D generative priors (writeup §2 / contribution 3 "modern-prior robotization").
Each prior produces an **asset bank** that the robotization layer
(`genedynamics/morphology/`) converts into simulatable soft bodies — the prior's
output only has to reduce to **voxel occupancy** on a fixed grid
(`morphology/robotize_common.py:finalize_from_voxelization`).

## Results — Table 3

The final prior cross-comparison (the single source of truth — final table, metric
definitions, and analysis) lives in
[`results/soft_robot/co_design/table3/TABLE3.md`](../../results/soft_robot/co_design/table3/TABLE3.md).

## Selection decision (2026-06-18): text-to-3D PRIMARY, image/mesh/3DGS pluggable

Derived from the method (see `docs/soft_robot_codesign_paper_plan.md`):
- **Gradient-free** MBD score consumes the prior only as a pointwise-evaluated
  occupancy field + a `log p₀` scalar — it is **never differentiated**. So
  differentiability is NOT a selection criterion (this frees us from DiffuseBot's
  hard differentiable-SDF-decoder constraint).
- **Apples-to-apples vs DiffuseBot**: DiffuseBot's default prior is *text*-conditioned
  Point-E (`base40M-textvec`), so a text-to-3D prior is the controlled-variable match.
- **Bank breadth**: text→bank→robotize needs no image-curation step (image-to-3D does).

So the selection axis collapses to: (a) easy diverse robot-morphology bank, (b) clean
reduction to occupancy. **Image-to-3D stays a drop-in adapter** (modality-agnostic
robotize tail), just off the main axis.

## Prior candidates — availability-verified (2026-06, code+weights checked)

The README list was a candidate shortlist; this is the **empirically-verified** status
after checking each repo/HF for actually-released **text** code+weights. The Table-3 text
axis (apples-to-apples vs DiffuseBot's text-conditioned Point-E) **locks to the 4 priors
that are genuinely text-native AND have public weights**, which conveniently span the
**4 major 3D representations** (point-cloud / implicit-SDF / structured-voxel / 3DGS) —
a stronger "robotize across representations" story than 8 same-flavor priors.

### LOCKED text axis (Table 3 rows) — wiring now

| Prior | Repr. | Source / weights | → occupancy | Status |
|---|---|---|---|---|
| **Point-E** | point cloud | OpenAI `point-e` (pip, MIT) | pc → `pc_robotize` → occ | ✅ text, DiffuseBot anchor |
| **Shap-E** | implicit SDF | OpenAI `shap-e` (pip, MIT) | implicit → mesh → occ | ✅ text-native, A2 basis |
| **TRELLIS-text** | structured voxel | [`microsoft/TRELLIS-text-xlarge`](https://huggingface.co/microsoft/TRELLIS-text-xlarge) (CVPR'25) | SLAT ≈ occ | ✅ text, occupancy-native |
| **SplatFlow** | 3D Gaussian | [`gohyojun15/SplatFlow`](https://github.com/gohyojun15/SplatFlow) + HF (CVPR'25) | GS → `gs_robotize` fill → occ | ✅ text→3DGS |
| **random_shapes** | procedural | in-repo (`priors/random_shapes.py`) | direct occ | ✅ baseline/control |

### Off the text axis (verified) — image arm or deferred

| Prior | Why off-axis (verified 2026-06) |
|---|---|
| **TripoSG** | **image**-only (single image). ✅ wired in G4 → use as the **image-arm** demo (modality-agnostic robotize tail). |
| **TRELLIS.2 / O-Voxel** | [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) released as **image-to-3D** (4B, ~H100). Open, but not text → image arm only. |
| **CraftsMan3D** | [`craftsman3d/craftsman`](https://huggingface.co/craftsman3d/craftsman) weights are **image/multiview**-primary → image arm only. |
| **Turbo3D** | ❌ code "coming soon" ([`hzhupku/Turbo3D`](https://github.com/hzhupku/Turbo3D) empty), **no weights** (Adobe). Deferred until release. |
| **DiffGS** | code has a text mode but only **chair-unconditional** weights are public — no text weights → would need training. Deferred. |
| **MeshFlow** | upstream not yet released (placeholder adapter); image/uncond, not text. Deferred. |
| **UVGS** | a GS **representation / A2 latent basis**, not a standalone text generator → not a Table-3 row. |

Notes:
- **Robotize cleanliness**: mesh/SDF priors (TRELLIS/Shap-E) are watertight solids → trivial
  robotize. **3DGS priors (SplatFlow) emit surface Gaussians (hollow shell)** → `gs_robotize`
  must density-fill a SOLID interior for MPM. Point-E emits a point cloud → `pc_robotize`.
- **4DGS / dynamic-gen: NOT used** — temporal priors are redundant with the co-designed
  controller φ and break the JIT fixed-particle fast path.
- **A2 synergy**: Shap-E / TRELLIS latent can seed the A2 morphology decoder `g(w)→occupancy`.
- **Table 3 columns**: 8 shape/connectivity/diversity/prior-NLL metrics (CPU, task-independent;
  `morphology/shape_metrics.py`) + 1 locomotion-reward column (GPU). 4-task generality is a
  Table-1 concern, decoupled from this prior table.

## Setup (offline bank generation)

Each heavy 3D-gen prior runs **offline in its own isolated venv** (deps conflict with the
JAX/MPM env — e.g. TripoSG pins `numpy<2`; see G4 in the runbook for the isolated-venv +
torchaudio-stub recipe). The online co-design loop only loads cached robotized specs, so the
prior never shares a GPU process with the MPM rollouts.

The prior repos are **git submodules** (pinned upstream commits), kept **pristine** so the
parent repo only records clean pointers:

```sh
git submodule update --init third_party/morphology_priors/{trellis_repo,splatflow_repo,triposg/repo}
# Shap-E / Point-E: pip install from OpenAI repos (no submodule). Follow each upstream README for weights.
```

> **Submodules are never hand-edited.** Per-prior integration code lives in the PARENT
> repo at `scripts/tasks/soft_robot/morphology/prior_adapters/{pointe,shape,trellis,splatflow,triposg}_batch_infer.py`
> (these `sys.path`-inject + `chdir` into the submodule, so the submodule stays clean).
> Where a loader genuinely needs an upstream-file edit (SplatFlow's fp16/`from_config`
> fix), it is captured as a **tracked patch** applied to the fresh checkout, NOT committed
> into the submodule:
> ```sh
> git -C third_party/morphology_priors/splatflow_repo apply \
>   scripts/tasks/soft_robot/morphology/prior_adapters/splatflow_fp16_fromconfig.patch
> ```

Bank generation: run the per-prior adapter (`prior_adapters/<name>_batch_infer.py`) in its
isolated venv → `data/asset_banks/<name>/{raw,robotized}/`. RobotizationSuccess + diversity
+ loco reward per prior feed **Table 3** (see SETUP.md for venv/path/cache details).

## Adapter contract

Each prior gets a lazy adapter at `genedynamics/morphology/priors/<name>.py`
(`register_lazy_mesh_prior` template) exposing `sample(prompt, n, seed) -> List[trimesh.Trimesh]`
(or a point-cloud/gaussian variant routed to `pc_robotize`/`gs_robotize`). The adapter hides
all per-prior config so the rest of the pipeline sees only a shape reducible to occupancy.
⚠️ adapters are **image-to-3D vs text-to-3D aware**: image priors (TripoSG, MeshFlow) need an
input image; text priors (Shap-E, TRELLIS-text, CraftsMan, Point-E) take a prompt directly.
