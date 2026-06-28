# Table-3 prior pipeline — reproducible setup & paths

Every heavy text-to-3D prior runs in its **own isolated venv** and dumps raw
assets; the MAIN genedynamics env robotizes them. The prior never shares a GPU
process with the JAX/MPM rollouts.

## Path map (this box)

| What | Path |
|---|---|
| Isolated venvs | `/workspace/morph_venvs/{pointe, shape, splatflow}` (`trellis` → symlink to `splatflow`) |
| Cache/env redirect | `/workspace/morph_cache/env.sh` (`TMPDIR`/`HF_HOME`/`TORCH_HOME`/`PIP_CACHE_DIR` → `/workspace`) |
| **Point-E / Shap-E weights** | **`data/model_caches/{point_e,shap_e}_model_cache`** (gitignored under `/data`; the batch_infer `--cache-dir` defaults here and `chdir`s so the CLIP `ViT-L-14.pt` lands there too — never the repo root) |
| TRELLIS weights (~6 GB) | `HF_HOME=/root/.cache/hf` — on the LOCAL `/` overlay, **not** `/workspace` |
| Prior repos | submodules `third_party/morphology_priors/{trellis_repo,splatflow_repo,triposg/repo}` (pristine) |
| Per-prior adapters (parent repo) | `scripts/tasks/soft_robot/morphology/prior_adapters/{pointe,shape,trellis,splatflow,triposg}_batch_infer.py` (+ `splatflow_fp16_fromconfig.patch`) |
| SplatFlow GS weights / SD3 fp16 | `data/splatflow_ckpts` (8.3 GB) / `/root/.cache/hf/hub` (15 GB, on `/`) |
| Shared text prompts | `configs/soft_robot/co_design/table3/prompts.txt` |
| Raw + robotized banks | `data/asset_banks/{pointe,shape,trellis,splatflow,loco_cpu,triposg}/{raw,robotized}/` |
| Results (table + per-prior json) | `results/soft_robot/co_design/table3/` |

> ⚠️ **`/workspace` has a per-user disk QUOTA** (a `dd` test write fails at ~2–3 GB
> free even though `df` shows 259 TB). Big model weights therefore go to the LOCAL
> `/` overlay (`HF_HOME=/root/.cache/hf`, ~18 GB free), while venvs stay on
> `/workspace`. Clear `/root/.cache/pip` (`pip cache purge`) if `/` gets tight.

## env redirect (`source` before any prior command)

```sh
# /workspace/morph_cache/env.sh
export TMPDIR=/workspace/morph_tmp
export HF_HOME=/workspace/morph_cache/hf      # override to /root/.cache/hf for TRELLIS
export TORCH_HOME=/workspace/morph_cache/torch
export PIP_CACHE_DIR=/workspace/morph_cache/pipcache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Per-prior install (verified 2026-06-18, RTX A5000, torch 2.10 main / 2.4 for TRELLIS)

### Point-E (point-cloud, text) — easiest
```sh
python -m venv --system-site-packages /workspace/morph_venvs/pointe   # reuse main torch 2.10
/workspace/morph_venvs/pointe/bin/pip install git+https://github.com/openai/point-e
# generate:
/workspace/morph_venvs/pointe/bin/python \
  third_party/morphology_priors/pointe/batch_infer.py \
  --prompts-file configs/soft_robot/co_design/table3/prompts.txt \
  --out-dir data/asset_banks/pointe/raw --n-per-prompt 3 --seed 0
```

### Shap-E (implicit-SDF, text) — easiest
```sh
python -m venv --system-site-packages /workspace/morph_venvs/shape
/workspace/morph_venvs/shape/bin/pip install git+https://github.com/openai/shap-e
# generate: same pattern, third_party/morphology_priors/shape/batch_infer.py
```

### TRELLIS-text (structured-voxel, text) — heavy but NO CUDA compiles
Reuses the 2.4 torch venv. Key tricks: vendored flexicubes is a **git submodule**;
sparse attention via **xformers** (avoid flash-attn); spconv + kaolin from **pip wheels**;
mesh is `MeshExtractResult.vertices/faces` (skip nvdiffrast/`to_glb`); weights to `/`.
```sh
cd third_party/morphology_priors
git clone --depth 1 https://github.com/microsoft/TRELLIS.git trellis_repo
cd trellis_repo && git submodule update --init --recursive   # pulls flexicubes
V=/workspace/morph_venvs/splatflow/bin       # torch 2.4.0+cu121 (also: morph_venvs/trellis)
$V/pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
$V/pip install spconv-cu120 kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
$V/pip install easydict einops safetensors transformers trimesh scipy igraph \
   rembg open3d sympy plotly pandas pillow imageio tqdm opencv-python-headless onnxruntime
$V/pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e
# generate (weights auto-download to /root/.cache/hf on first run):
HF_HOME=/root/.cache/hf ATTN_BACKEND=xformers SPCONV_ALGO=native TMPDIR=/tmp $V/python \
  third_party/morphology_priors/trellis_repo/batch_infer.py \
  --prompts-file configs/soft_robot/co_design/table3/prompts.txt \
  --out-dir data/asset_banks/trellis/raw --n-per-prompt 3 --seed 0
```

### SplatFlow (3D-Gaussian, text) — WORKING (2026-06-18)
Needs a HF token with the **gated** `stabilityai/stable-diffusion-3-medium-diffusers`
license accepted (web). Venv `/workspace/morph_venvs/splatflow` (torch 2.4 + pytorch3d +
diffusers 0.30). SplatFlow GS weights (`gs_decoder`+`mv_rf_ema`, 8.3 GB) → `data/splatflow_ckpts`;
SD3 fp16 weights (15 GB) → `/root/.cache/hf/hub` (local `/`, /workspace quota can't hold 15 GB).
```sh
HF_TOKEN=<tok> hf download stabilityai/stable-diffusion-3-medium-diffusers \
  --include "*.fp16.safetensors" "*.json" "*.model" "*.txt" --cache-dir /root/.cache/hf  # see ⚠️ path note
# apply the loader fix to the pristine submodule (variant=fp16 + transformer from_config):
git -C splatflow_repo apply scripts/tasks/soft_robot/morphology/prior_adapters/splatflow_fp16_fromconfig.patch
HF_HOME=/root/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $V/python \
  scripts/tasks/soft_robot/morphology/prior_adapters/splatflow_batch_infer.py \
  --prompts-file configs/soft_robot/co_design/table3/prompts.txt \
  --out-dir data/asset_banks/splatflow/raw --ckpt-dir data/splatflow_ckpts --n-per-prompt 3 --num-steps 50
```
**NOT a VRAM problem** (full model = 15.5/24 GB). Three real blockers, all fixed:
1. **HF cache-path mismatch (main culprit):** `snapshot_download(cache_dir=X)` stores at
   `X/models--…` but `from_pretrained(HF_HOME=X)` reads `X/hub/models--…` → it never finds
   the weights → re-downloads forever → fills the 30 GB `/` overlay (the bogus "No space").
   Fix: keep them under `…/hub/` (or `mv` them there).
2. **variant mismatch:** only `.fp16` shards downloaded → loaders default to non-fp16 names.
   Patch adds `variant="fp16"`+`use_safetensors=True`; MV transformer → `from_config` (skips a
   4 GB download, weights are overwritten by `mv_rf_ema` anyway). Captured in the `.patch`.
3. **`cfg=false` → `clean_ray_latent` UnboundLocalError** (it's only assigned in the CFG
   branch). Adapter runs `cfg=true` + `sd3_guidance=false` (drops the 4 GB SD3 transformer)
   + skips the step-2 refiner.

**Gaussian → robotize:** SplatFlow emits ~524 k gaussians, ~94 % near-transparent background +
far floaters. `build_bank_from_raw.py` centers-branch keeps opacity>0.5, trims spatial outliers
(dist < 92nd pct), then `robotize_point_cloud` (voxelize + solid-fill). Result: **30/30 robotize,
gen 41.5 s/asset, loco-R −3.01** (highest fidelity, worst locomotor).

## MAIN-env robotize + metrics (no venv)

```sh
# raw mesh/pc/gaussian -> robotized SoftBodySpec on the 3x3x3 grid
python scripts/tasks/soft_robot/morphology/build_bank_from_raw.py \
  --raw-dir data/asset_banks/<prior>/raw --bank-root data/asset_banks/<prior> \
  --voxel-dims 3,3,3 --n-actuators 10
# shape columns (CPU) + loco-reward column (GPU) -> assemble Table 3
python scripts/tasks/soft_robot/co_design/analysis/table3_loco_reward.py --bank-dir <bank> --prior-name <p> --out results/.../<p>_loco_reward.json
python scripts/tasks/soft_robot/co_design/analysis/table3_assemble.py --priors <p>:<bank> ... --reward-dir results/soft_robot/co_design/table3
```

## Gotchas hit (so the next person doesn't)

- **flexicubes** is a TRELLIS git submodule — `--depth 1` clone skips it → `git submodule update --init`.
- TRELLIS clean venv is missing many BASIC deps (rembg/open3d/sympy/plotly/pandas/kaolin); `trellis.pipelines` imports them all at load.
- `/workspace` quota → weights to `/` (HF_HOME=/root/.cache/hf).
- MFS-slow imports → use ≥400 s timeouts when probing; runtime model-load is a one-time cost.
- `prompts.txt` starts with comment lines → slice prompts with `grep -vE '^#|^$' | head -N`, not `head -N`.
- **Point-E/Shap-E cache pollution**: both libs hard-code `default_cache_dir() == CWD/<name>_model_cache`
  (used by `load_checkpoint` AND the CLIP `ViT-L-14` text encoder), which dumps ~5 GB into the repo
  root. Fixed in `batch_infer.py` by `chdir`-ing into `data/model_caches/` so every download lands
  under the gitignored `data/`. Anything data-related stays under `data/`.
