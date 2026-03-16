# 3DGS Config Layout (Phase 0 Cleanup)

This folder is now organized by experiment role instead of historical file names.

## Structure

- `main/`: canonical method runs (paper/mainline behavior)
- `ablations/`: controlled switches (iid vs corr, warm-start, etc.)
- `baselines/`: non-method baselines (gsplat training, 3DGS-MAP protocol)
- `_template.yaml`: template for creating new configs
- `_base_nerf_synth.yaml`: shared reference defaults

## Current configs

### Main

- `main/lego_mbd_canonical.yaml`

### Ablations

- `ablations/lego_mbd_iid_ablation.yaml`
- `ablations/lego_mbd_corr_ablation.yaml`
- `ablations/lego_mbd_warmstart_ablation.yaml`
- `ablations/chair_mbd_iid_ablation.yaml`
- `ablations/chair_mbd_corr_ablation.yaml`

### Baselines

- `baselines/lego_gsplat_baseline.yaml`
- `baselines/lego_gsplat_densify_baseline.yaml`
- `baselines/lego_3dgs_map_baseline.yaml`

## Run examples

```bash
# Canonical MBD run (no warm start, ~5 dB PSNR)
python scripts/tasks/3dgs/run_full_experiment.py configs/3dgs/main/lego_mbd_canonical.yaml

# PSNR-focused: warm start from gsplat (~10 dB PSNR)
# Step 1: Train gsplat for warm start (128 gaussians, ~20 dB on train views)
python scripts/tasks/3dgs/train_gsplat.py configs/3dgs/main/lego_mbd_canonical.yaml \
  --output results/3dgs/lego_gsplat_warmstart --iters 6000 --n-gaussians 128
# Step 2: Run MBD with prior-centered init (Phase 3+4: best_chain, train/test metrics)
python scripts/tasks/3dgs/run_full_experiment.py configs/3dgs/main/lego_mbd_quality_recovery_vramfit.yaml \
  --initial-scene-path results/3dgs/lego_gsplat_warmstart/scene_params.npz \
  --initialization-mode prior_center --init-jitter-scale 0.1 --n-seeds 2 --best-chain

# One-liner: bash scripts/tasks/3dgs/run_quality_recovery.sh

# Quality recovery configs (tuned for PSNR)
python scripts/tasks/3dgs/run_full_experiment.py configs/3dgs/main/lego_mbd_quality_recovery_vramfit.yaml
python scripts/tasks/3dgs/run_full_experiment.py configs/3dgs/main/lego_mbd_psnr_improved.yaml  # 192 gaussians, may OOM on smaller GPUs

# Likelihood ablation
python scripts/tasks/3dgs/run_full_experiment.py configs/3dgs/ablations/lego_mbd_iid_ablation.yaml
python scripts/tasks/3dgs/run_full_experiment.py configs/3dgs/ablations/lego_mbd_corr_ablation.yaml

# Baseline (pure gsplat training path)
python scripts/tasks/3dgs/run_baseline_experiment.py configs/3dgs/baselines/lego_gsplat_baseline.yaml --iters 30000
python scripts/tasks/3dgs/run_baseline_experiment.py configs/3dgs/baselines/lego_gsplat_densify_baseline.yaml --iters 30000 --densify --max-gaussians 30000
```

## PSNR improvement

| Setup | PSNR | LPIPS |
|-------|------|-------|
| Canonical (random init) | ~5.4 dB | ~0.98 |
| Quality recovery vramfit | ~5.4 dB | ~0.97 |
| **Warm start** (gsplat 6k iters → MBD prior_center) | **~10.3 dB** | **~0.76** |

Warm start: train gsplat 6k iters with 128 gaussians (~20 dB on train), then run MBD with `--initial-scene-path` and `--init-jitter-scale 0.1`.

## Notes

- The `gsplat` baselines are intentionally separated from `main` to avoid mixing method and baseline semantics.
- Warm-start behavior is treated as an ablation, not the canonical method definition.
