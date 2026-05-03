# 3D Foundation Morphology Priors

Stub directory for the modern 3D generative priors used in writeup §2 (TripoSG
as the main prior; TRELLIS / Hunyuan3D as ablations). These produce the
**asset bank** that the robotization layer (`genedynamics/morphology/`)
converts into simulatable soft bodies.

Each prior lives in its own subdir and is **not** imported until Phase 3.
Phase 0 / 1 only needs the directory to exist so adapter modules know where
to point.

## Subdirectories

| Dir | Source | Status | Used in |
|---|---|---|---|
| `triposg/` | https://github.com/VAST-AI-Research/TripoSG | required for Phase 3 | main prior — image/text → mesh |
| `trellis/` | https://github.com/microsoft/TRELLIS | optional ablation | structured-latent prior |
| `hunyuan3d/` | https://github.com/Tencent/Hunyuan3D-2 | optional ablation | high-res mesh asset prior |

## Setup (Phase 3 onward)

```sh
cd third_party/morphology_priors
git clone https://github.com/VAST-AI-Research/TripoSG.git triposg
# follow upstream README for model weight download
```

Asset bank generation is **offline** (`scripts/morphology/build_asset_bank.py`,
to be added in Phase 3) and writes meshes + metadata to
`data/asset_banks/<prior_name>/`. The online co-design loop only loads cached
meshes, so the heavy 3D prior never has to live in the same GPU process as
the MPM rollouts.

## What our adapters expect

Each prior gets a thin adapter at `genedynamics/morphology/priors/<name>.py`
exposing `sample_assets(prompts, n_per_prompt) -> List[trimesh.Trimesh]`. The
adapter handles all per-prior config (sampler, guidance scale, decoding
format) so the rest of the pipeline sees only meshes.
