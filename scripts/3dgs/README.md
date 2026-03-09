# 3DGS 实验脚本清单

## 脚本一览

| 脚本 | 用途 | 参数 |
|------|------|------|
| `run_full_experiment.py` | 完整实验：多 seed 后验 + 指标 + 图表 | `config [--n-seeds 8]` |
| `run_all.sh` | 批量运行 lego+chair iid/corr | - |
| `run_3dgs_map.sh` | 3DGS-MAP baseline（官方 3DGS 训练） | `[dataset] [output]` |
| `run_mbd3d_iid.sh` | Ours-MBD (iid) | `[object]` 默认 lego |
| `run_mbd3d_corr.sh` | Ours-MBD (corr) | `[object]` 默认 lego |
| `eval_3dgs_metrics.py` | 评估 PSNR/LPIPS/NLL | `result_dir [--split test] [--output path]` |
| `export_3dgs_figures.py` | 导出对比图、不确定性热图 | `result_dir [--output dir]` |

## 用法

```bash
# 完整实验（多 seed 后验、PSNR/LPIPS/NLL、定性图 + 不确定性热图）
python scripts/3dgs/run_full_experiment.py configs/3dgs/lego_mbd_iid.yaml --n-seeds 8

# 批量运行所有 Ours-MBD 实验
./scripts/3dgs/run_all.sh

# 3DGS-MAP baseline（需官方 3DGS 仓库）
./scripts/3dgs/run_3dgs_map.sh data/nerf_synthetic/lego results/3dgs/lego_3dgs_map

# Ours-MBD (iid)
./scripts/3dgs/run_mbd3d_iid.sh lego

# Ours-MBD (corr)
./scripts/3dgs/run_mbd3d_corr.sh lego

# 评估指标
python scripts/3dgs/eval_3dgs_metrics.py results/3dgs/lego_mbd_iid --output results/3dgs/lego_mbd_iid/metrics.json

# 导出图表
python scripts/3dgs/export_3dgs_figures.py results/3dgs/lego_mbd_iid --output results/3dgs/lego_mbd_iid/figures
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `VENV` | Python 路径，默认 `.venv_arm64/bin/python` |
| `GAUSSIAN_SPLATTING_PATH` | 官方 3DGS 仓库路径 |
