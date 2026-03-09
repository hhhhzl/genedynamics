# 3DGS NeRF Synthetic Experiments (Experiment 1: Small object 3D reconstruction)

Aligned with 3DGS setting for motivation experiment.

## Configurations

| File | Method | Object |
|------|--------|--------|
| `_template.yaml` | 模板（复制后修改） | - |
| `_base_nerf_synth.yaml` | Base (override in children) | - |
| `lego_mbd_iid.yaml` | Ours-MBD (iid) | lego |
| `lego_mbd_corr.yaml` | Ours-MBD (corr) | lego |
| `lego_3dgs_map.yaml` | 3DGS-MAP baseline | lego |
| `chair_mbd_iid.yaml` | Ours-MBD (iid) | chair |
| `chair_mbd_corr.yaml` | Ours-MBD (corr) | chair |

## 新增配置

复制 `_template.yaml` 为 `<object>_mbd_<iid|corr>.yaml`，修改 `name`、`output_dir`、`env_params.dataset_root`、`env_params.object`、`method_params.use_lowrank_noise`。

## Data

- **Dataset**: NeRF Synthetic (transforms_train/val/test.json)
- **Objects**: lego, chair, drums, ficus, hotdog, materials, ship
- **Views**: 50–100 (max_views)
- **Resolution**: infer 128x128, eval 512x512

## Dataset layout

```
data/nerf_synthetic/
  lego/
    transforms_train.json
    transforms_val.json
    transforms_test.json
    train/
      r_*.png
    val/
    test/
  chair/
    ...
```

Download from [NeRF Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

## Run

```bash
# Ours-MBD (iid) on lego
python -m genedynamics.experiments.runner configs/3dgs/lego_mbd_iid.yaml

# 3DGS-MAP baseline (requires official 3DGS repo)
./scripts/3dgs/run_3dgs_map.sh data/nerf_synthetic/lego results/3dgs/lego_3dgs_map
```

## Metrics

- held-out PSNR / LPIPS
- NLL (corr only)
- Inference budget (render passes to reach PSNR)

## PSNR expectations

- **MockRenderer** (default on CPU/Mac): surrogate renderer for optimization; PSNR ~9 dB (not meaningful).
- **GsplatRenderer** (CUDA): real 3DGS rendering for evaluation. On CUDA machines, `run_full_experiment.py` auto-uses GsplatRenderer when available for realistic PSNR (typically 25–35 dB on NeRF Synthetic).
- For ideal PSNR, run on a CUDA machine with `pip install gsplat`.
