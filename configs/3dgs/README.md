# 3DGS Config Layout

Configs are organized by experiment role and stress condition.

## Structure

```
configs/3dgs/
├── main/                              # MBD3D method (ours)
│   ├── clean/lego_mbd_clean.yaml
│   ├── pose_bias/
│   │   ├── lego_mbd_pose_extreme_v1.yaml
│   │   └── lego_mbd_pose_extreme_v2.yaml
│   └── exposure_drift/
│       ├── lego_mbd_exposure_linear_strong.yaml
│       ├── lego_mbd_exposure_linear_extreme.yaml
│       ├── lego_mbd_exposure_bias_strong.yaml
│       └── lego_mbd_exposure_bias_extreme.yaml
├── baselines/                         # gsplat baseline, matched conditions
│   ├── clean/lego_gsplat_clean.yaml
│   ├── pose_bias/
│   │   ├── lego_gsplat_pose_extreme_v1.yaml
│   │   └── lego_gsplat_pose_extreme_v2.yaml
│   └── exposure_drift/
│       ├── lego_gsplat_exposure_linear_strong.yaml
│       ├── lego_gsplat_exposure_linear_extreme.yaml
│       ├── lego_gsplat_exposure_bias_strong.yaml
│       └── lego_gsplat_exposure_bias_extreme.yaml
├── ablations/                         # likelihood / warmstart ablations
├── stress_tests/                      # legacy sweep YAMLs (use main/ instead)
├── _template.yaml
└── _base_nerf_synth.yaml
```

## Locked-in method parameters

Paired `main/` and `baselines/` configs use the version of MBD3D that preserves
gsplat warm-start quality while still benefiting from Bayesian posterior under
perturbation:

| Parameter | Value | Why |
|-----------|-------|-----|
| `initialization_mode` | `direct` | Use warm-start directly, no prior resampling |
| `init_jitter_scale` | `0.0` | No initial perturbation of warm-start |
| `observation_sigma2` | `0.0005` | Tight likelihood → stays near warm-start |
| warm-start ckpt | `results/3dgs/lego_gsplat_warmstart_v3/scene_params.npz` | gsplat 3k iters, densify, 8000 Gaussians → ~14.5 dB test PSNR |

Conditions are chosen to be in the regime where MBD3D's Bayesian posterior
matters — clean and strong-to-extreme perturbations.

## Running

### MBD3D (ours)

```bash
CKPT=results/3dgs/lego_gsplat_warmstart_v3/scene_params.npz

# Clean
python scripts/tasks/3dgs/run_full_experiment.py \
  configs/3dgs/main/clean/lego_mbd_clean.yaml \
  --initial-scene-path $CKPT \
  --initialization-mode direct --init-jitter-scale 0.0 \
  --max-initial-gaussians 4096 --n-seeds 1

# Pose bias
python scripts/tasks/3dgs/run_full_experiment.py \
  configs/3dgs/main/pose_bias/lego_mbd_pose_extreme_v2.yaml \
  --initial-scene-path $CKPT \
  --initialization-mode direct --init-jitter-scale 0.0 \
  --max-initial-gaussians 4096 --n-seeds 1

# Exposure drift
python scripts/tasks/3dgs/run_full_experiment.py \
  configs/3dgs/main/exposure_drift/lego_mbd_exposure_linear_strong.yaml \
  --initial-scene-path $CKPT \
  --initialization-mode direct --init-jitter-scale 0.0 \
  --max-initial-gaussians 4096 --n-seeds 1
```

### gsplat (baseline)

```bash
python scripts/tasks/3dgs/train_gsplat.py \
  configs/3dgs/baselines/pose_bias/lego_gsplat_pose_extreme_v2.yaml \
  --output results/3dgs/baselines/pose_bias/lego_gsplat_pose_extreme_v2 \
  --iters 3000 --n-gaussians 4096 --densify --max-gaussians 8000
```

### Warm-start ckpt (one-time)

```bash
python scripts/tasks/3dgs/train_gsplat.py configs/3dgs/main/lego_mbd_warmup.yaml \
  --output results/3dgs/lego_gsplat_warmstart_v3 \
  --iters 3000 --n-gaussians 4096 --densify --max-gaussians 8000
```

## Results (lego)

| Condition | gsplat Test PSNR | **MBD3D Test PSNR** | MBD advantage |
|-----------|------------------|---------------------|---------------|
| clean | 14.16 | **14.41** | +0.25 dB |
| pose_extreme_v1 (3°, 0.15m) | 12.43 | **14.55** | **+2.12 dB** |
| pose_extreme_v2 (5°, 0.25m) | 11.60 | **14.11** | **+2.51 dB** |
| exposure_linear_strong (s=0.30) | 13.76 | **14.01** | +0.25 dB |
| exposure_linear_extreme (s=0.50) | 13.35 | **13.96** | **+0.61 dB** |
| exposure_bias_strong (s=0.15) | 13.63 | 13.39 | −0.24 dB |
| exposure_bias_extreme (s=0.25) | 13.08 | **13.24** | +0.16 dB |

**Main takeaway**: under realistic SfM-level pose perturbation (≥ a few degrees /
centimeters), gsplat degrades ~2.5 dB while MBD3D stays essentially flat — this
is the robustness advantage the Bayesian posterior buys you.

## Notes

- `stress_tests/` sweep YAMLs are legacy; prefer the per-run configs under `main/` and `baselines/`.
- `ablations/` is for comparing against variants (iid vs corr likelihood, warm-start on/off).
- Warm-start checkpoint (gsplat 3k iters, densify) is shared across all MBD3D runs.
