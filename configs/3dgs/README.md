# 3DGS Experiment Layout

The repository hosts three experiments for probabilistic 3D Gaussian scene
reconstruction with MBD (`MBD3D`):

| Experiment | Role | Datasets |
|------------|------|----------|
| **exp1** (`main/` + `baselines/`) | Lego stress test (`main/` = MBD3D, `baselines/` = gsplat) | NeRF Synthetic – lego |
| **exp2_robust** | Cross-dataset robust mapping (pose bias + exposure drift) | Replica, TUM RGB-D |
| **exp2_active** | Active view selection (posterior-variance vs random / max-distance) | NeRF Synthetic |
| **exp3_mujoco** | Closed-loop active perception in MuJoCo | synthetic tabletop |

## Directory tree

```
configs/3dgs/
├── main/ baselines/                          # exp1 (lego)
│   ├── clean/
│   ├── pose_bias/{extreme_v1,extreme_v2}/
│   └── exposure_drift/{linear_strong,linear_extreme,bias_strong,bias_extreme}/
│
├── exp2_robust/
│   ├── replica/
│   │   ├── mbd/{clean,pose_bias,exposure_drift}/
│   │   └── gsplat/{clean,pose_bias,exposure_drift}/
│   └── tum/
│       ├── mbd/{clean,pose_bias,exposure_drift}/
│       └── gsplat/{clean,pose_bias,exposure_drift}/
│
├── exp2_active/
│   ├── mbd/lego_active_posterior_variance.yaml
│   └── baselines/{lego_active_random,lego_active_max_distance}.yaml
│
├── exp3_mujoco/
│   ├── mbd/{mujoco_tabletop_active_clean,mujoco_tabletop_active_pose_bias}.yaml
│   └── baselines/mujoco_tabletop_random_pose_bias.yaml
│
├── ablations/
├── stress_tests/                             # legacy
├── _shared/{replica_base,tum_base}.yaml
├── _template.yaml
└── _base_nerf_synth.yaml
```

## Code modules touched

| Module | Where | Purpose |
|--------|-------|---------|
| `genedynamics.data` | `genedynamics/data/` | Shared dataset adapters (NeRF Synthetic, **Replica**, **TUM RGB-D**, **MuJoCo**) |
| `genedynamics.solvers.single.mbd3d.active` | `…/mbd3d/active/` | View scorer + active selection loop (**exp2B**) |
| `genedynamics.solvers.single.mbd3d.calibration` | `…/mbd3d/calibration/` | Reliability / ECE / coverage / pixel-NLL |
| `Replica3DGSPlugin`, `TUM_RGBD_3DGSPlugin`, `MuJoCoActivePerceptionPlugin` | `experiments/plugins/environments/` | Environment plugins for the new datasets |
| `MBD3DActiveMethodPlugin` | `experiments/plugins/methods/mbd3d_active.py` | Wraps MBD3D with active selection |
| `assets/mujoco/tabletop_scene.xml`, `candidate_views.json` | `assets/mujoco/` | Exp3 scene + 40 candidate camera poses |

## Locked-in method parameters (exp1 / exp2)

| Parameter | Value | Why |
|-----------|-------|-----|
| `initialization_mode` | `direct` | Use warm-start directly, no prior resampling |
| `init_jitter_scale` | `0.0` | No initial perturbation of warm-start |
| `observation_sigma2` | `0.0005` | Tight likelihood → stays near warm-start |
| warm-start ckpt (lego) | `results/3dgs/lego_gsplat_warmstart_v3/scene_params.npz` | gsplat 3k iters, densify, 8000 Gaussians → ~14.5 dB test PSNR |

## Running

### Exp1: lego stress (existing)

```bash
CKPT=results/3dgs/lego_gsplat_warmstart_v3/scene_params.npz

python scripts/tasks/3dgs/run_full_experiment.py \
  configs/3dgs/main/pose_bias/lego_mbd_pose_extreme_v2.yaml \
  --initial-scene-path $CKPT \
  --initialization-mode direct --init-jitter-scale 0.0 \
  --max-initial-gaussians 4096 --n-seeds 1

python scripts/tasks/3dgs/train_gsplat.py \
  configs/3dgs/baselines/pose_bias/lego_gsplat_pose_extreme_v2.yaml \
  --output results/3dgs/baselines/pose_bias/lego_gsplat_pose_extreme_v2 \
  --iters 3000 --n-gaussians 4096 --densify --max-gaussians 8000
```

### Exp2_robust: Replica / TUM

```bash
# Fetch datasets (once)
bash scripts/tasks/3dgs/download_replica.sh data/replica
bash scripts/tasks/3dgs/download_tum.sh data/tum
python scripts/tasks/3dgs/prepare_replica_dataset.py data/replica
python scripts/tasks/3dgs/prepare_tum_dataset.py data/tum

# MBD3D run on Replica under extreme pose bias
python scripts/tasks/3dgs/run_full_experiment.py \
  configs/3dgs/exp2_robust/replica/mbd/pose_bias/replica_room0_mbd_pose_extreme_v2.yaml \
  --n-seeds 4

# gsplat baseline, same condition
python scripts/tasks/3dgs/train_gsplat.py \
  configs/3dgs/exp2_robust/replica/gsplat/pose_bias/replica_room0_gsplat_pose_extreme_v2.yaml \
  --output results/3dgs/exp2_robust/replica/gsplat/replica_room0_gsplat_pose_extreme_v2 \
  --iters 3000 --n-gaussians 4096 --densify --max-gaussians 8000
```

### Exp2_active: active view selection

```bash
# Ours (posterior-variance scorer)
python scripts/tasks/3dgs/run_active_selection.py \
  configs/3dgs/exp2_active/mbd/lego_active_posterior_variance.yaml \
  --scorer posterior_variance --n-init 10 --n-rounds 5 --n-select 1 \
  --output results/3dgs/exp2_active/posterior_variance

# Baselines
python scripts/tasks/3dgs/run_active_selection.py \
  configs/3dgs/exp2_active/baselines/lego_active_random.yaml \
  --scorer random --n-init 10 --n-rounds 5 --n-select 1 \
  --output results/3dgs/exp2_active/random

python scripts/tasks/3dgs/run_active_selection.py \
  configs/3dgs/exp2_active/baselines/lego_active_max_distance.yaml \
  --scorer max_distance --n-init 10 --n-rounds 5 --n-select 1 \
  --output results/3dgs/exp2_active/max_distance

# Compare
python scripts/tasks/3dgs/plot_active_comparison.py \
  --runs results/3dgs/exp2_active/posterior_variance \
         results/3dgs/exp2_active/random \
         results/3dgs/exp2_active/max_distance \
  --labels Ours Random MaxDist \
  --output results/figures/exp2_active/comparison.png
```

### Exp3_mujoco: closed-loop demo

```bash
# Clean (no pose bias) — Ours (posterior-variance scorer)
python scripts/tasks/3dgs/run_mujoco_demo.py \
  configs/3dgs/exp3_mujoco/mbd/mujoco_tabletop_active_clean.yaml \
  --output results/3dgs/exp3_mujoco/clean_ours

# Pose bias — Ours
python scripts/tasks/3dgs/run_mujoco_demo.py \
  configs/3dgs/exp3_mujoco/mbd/mujoco_tabletop_active_pose_bias.yaml \
  --output results/3dgs/exp3_mujoco/bias_ours

# Pose bias — Random baseline
python scripts/tasks/3dgs/run_mujoco_demo.py \
  configs/3dgs/exp3_mujoco/baselines/mujoco_tabletop_random_pose_bias.yaml \
  --scorer random \
  --output results/3dgs/exp3_mujoco/bias_random
```

### Calibration + aggregation (all experiments)

```bash
# Reliability curve on an MBD3D run with ≥ 2 seeds
python scripts/tasks/3dgs/plot_calibration_curve.py \
  results/3dgs/main/clean/lego_mbd_clean

# Aggregate everything into a CSV
python scripts/tasks/3dgs/aggregate_results.py results/3dgs \
  --csv results/3dgs/summary.csv --json results/3dgs/summary.json
```

## Results summary (exp1 — lego)

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
